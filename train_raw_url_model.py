"""Train and evaluate the PROM02 production raw-URL classifier.

This is intentionally separate from ``train_deployable.py``. The latter is a
historic UCI-feature benchmark; this command uses raw URLs and the same
``extract_raw_url_features`` function used by the app and webhook.

Example:
    python train_raw_url_model.py --input data/restricted/raw_url_input.csv \
        --output-dir artifacts/raw_url
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from raw_url_data import domain_disjoint_split, prepare_raw_url_data
from url_features import RAW_URL_FEATURES, extract_raw_url_features

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42


def feature_frame(urls: pd.Series) -> pd.DataFrame:
    """Return production features in the frozen, explicitly checked order."""
    return pd.DataFrame([extract_raw_url_features(url) for url in urls], columns=RAW_URL_FEATURES)


def model_specs(quick: bool) -> dict[str, tuple[Pipeline, dict]]:
    """The five classifiers named in the proposal, each with a modest honest grid."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # fail instead of silently replacing the proposed classifier
        raise RuntimeError("xgboost is required for the proposed five-model comparison; run pip install -r requirements.txt") from exc
    forest_trees = [80] if quick else [200, 400]
    xgb_trees = [80] if quick else [200, 350]
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
            {"clf__C": [0.3, 1.0] if quick else [0.1, 0.3, 1.0, 3.0]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED, class_weight="balanced"))]),
            {"clf__max_depth": [5, 10] if quick else [4, 8, 12, None], "clf__min_samples_leaf": [1, 4] if quick else [1, 3, 8]},
        ),
        "Random Forest": (
            Pipeline([("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced"))]),
            {"clf__n_estimators": forest_trees, "clf__max_depth": [10, None], "clf__min_samples_leaf": [1, 3]},
        ),
        "SVM (RBF)": (
            Pipeline([("scale", StandardScaler()), ("clf", SVC(probability=True, random_state=SEED, class_weight="balanced"))]),
            {"clf__C": [0.5, 2.0] if quick else [0.1, 0.5, 1.0, 2.0, 5.0], "clf__gamma": ["scale"]},
        ),
        "XGBoost": (
            Pipeline([("clf", XGBClassifier(
                random_state=SEED, eval_metric="logloss", n_jobs=-1,
                objective="binary:logistic", tree_method="hist",
            ))]),
            {"clf__n_estimators": xgb_trees, "clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]},
        ),
    }


def probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} has no probability interface")
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def choose_threshold(labels: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    """Optimise F1 on development calibration data only, preferring 0.5 on ties."""
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    values = [(float(threshold), float(f1_score(labels, scores >= threshold, zero_division=0))) for threshold in candidates]
    best_f1 = max(value[1] for value in values)
    threshold = min((value for value in values if value[1] == best_f1), key=lambda value: abs(value[0] - 0.5))[0]
    return threshold, best_f1


def metrics(labels: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def bootstrap_intervals(labels: pd.Series, scores: np.ndarray, threshold: float, iterations: int, seed: int) -> dict[str, list[float]]:
    """Non-parametric test-set bootstrap intervals; no resample influences training."""
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels, dtype=int)
    collected: dict[str, list[float]] = {key: [] for key in metrics(labels, scores, threshold)}
    for _ in range(iterations):
        indices = rng.integers(0, len(labels_array), len(labels_array))
        sample_y = labels_array[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        sample = metrics(pd.Series(sample_y), scores[indices], threshold)
        for key, value in sample.items():
            collected[key].append(value)
    return {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for key, values in collected.items() if values}


def write_explanations(model: Pipeline, features: pd.DataFrame, labels: pd.Series, output_dir: Path) -> str:
    """Write SHAP importances for supported models, otherwise permutation fallback."""
    sample = features.iloc[: min(300, len(features))]
    try:
        import shap

        estimator = model.named_steps["clf"]
        transformed = model[:-1].transform(sample) if len(model.steps) > 1 else sample
        if isinstance(estimator, (RandomForestClassifier, DecisionTreeClassifier)) or estimator.__class__.__name__ == "XGBClassifier":
            values = shap.TreeExplainer(estimator)(transformed)
        elif isinstance(estimator, LogisticRegression):
            values = shap.LinearExplainer(estimator, transformed)(transformed)
        else:
            raise TypeError("SVM has no robust fast SHAP explainer in this project configuration")
        array = np.asarray(values.values)
        if array.ndim == 3:
            array = array[:, :, 1] if array.shape[-1] == 2 else array[:, :, 0]
        importance = np.mean(np.abs(array), axis=0)
        pd.DataFrame({"feature": RAW_URL_FEATURES, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False).to_csv(output_dir / "shap_feature_importance.csv", index=False)
        (output_dir / "explanation_method.txt").write_text("SHAP values on a held-out raw-URL feature sample.\n", encoding="utf-8")
        return "shap"
    except Exception as exc:
        # The fallback is explicit, saved, and never called SHAP.
        result = permutation_importance(model, sample, labels.iloc[: len(sample)], n_repeats=10, random_state=SEED, n_jobs=-1, scoring="f1")
        pd.DataFrame({"feature": RAW_URL_FEATURES, "mean_importance": result.importances_mean, "std_importance": result.importances_std}).sort_values("mean_importance", ascending=False).to_csv(output_dir / "permutation_feature_importance.csv", index=False)
        (output_dir / "explanation_method.txt").write_text(f"Permutation-importance fallback; SHAP was unavailable for the selected estimator: {exc}\n", encoding="utf-8")
        return "permutation_importance_fallback"


def _split_development(development: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reserve groups for threshold selection after cross-validated model choice."""
    fit, calibration, manifest = domain_disjoint_split(development, test_size=0.25, random_state=SEED + 700, candidates=150)
    return fit, calibration, manifest


def train(input_path: Path, output_dir: Path, quick: bool, bootstrap_iterations: int) -> dict:
    raw = pd.read_csv(input_path)
    prepared, audit = prepare_raw_url_data(raw)
    synthetic_input = prepared["source"].str.contains("SYNTHETIC", case=False, na=False).any()
    development, locked_test, outer_manifest = domain_disjoint_split(prepared, test_size=0.2, random_state=SEED, candidates=200)
    fit, calibration, calibration_manifest = _split_development(development)
    if min(fit["label"].value_counts()) < 10 or fit["registered_domain"].nunique() < 5:
        raise ValueError("not enough grouped development data for five-fold CV; collect more independent domains")

    x_fit, y_fit = feature_frame(fit["url"]), fit["label"]
    x_cal, y_cal = feature_frame(calibration["url"]), calibration["label"]
    x_test, y_test = feature_frame(locked_test["url"]), locked_test["label"]
    n_splits = min(5, int(fit["registered_domain"].nunique()))
    cv = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(x_fit, y_fit, fit["registered_domain"]))

    rows, fitted = [], {}
    for name, (pipeline, grid) in model_specs(quick).items():
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, error_score="raise")
        started = time.perf_counter()
        search.fit(x_fit, y_fit)
        train_seconds = time.perf_counter() - started
        fitted[name] = search.best_estimator_
        calibration_scores = probabilities(search.best_estimator_, x_cal)
        threshold, calibration_f1 = choose_threshold(y_cal, calibration_scores)
        test_started = time.perf_counter()
        test_scores = probabilities(search.best_estimator_, x_test)
        latency_ms = (time.perf_counter() - test_started) * 1000 / max(len(x_test), 1)
        row = {
            "model": name,
            "cv_f1_mean": float(search.best_score_),
            "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "calibration_threshold": threshold,
            "calibration_f1": calibration_f1,
            "train_time_seconds": train_seconds,
            "latency_ms_per_url": latency_ms,
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            **metrics(y_test, test_scores, threshold),
        }
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(["cv_f1_mean", "model"], ascending=[False, True]).reset_index(drop=True)
    comparison.to_csv(output_dir / "model_comparison_raw_url.csv", index=False)
    selected_name = str(comparison.iloc[0]["model"])
    selected = fitted[selected_name]
    selected_threshold = float(comparison.iloc[0]["calibration_threshold"])
    selected_scores = probabilities(selected, x_test)
    selected_predictions = (selected_scores >= selected_threshold).astype(int)
    error_rows = locked_test.copy()
    error_rows["phishing_probability"] = selected_scores
    error_rows["prediction"] = selected_predictions
    error_rows["error_type"] = np.where(
        (error_rows["label"] == 0) & (error_rows["prediction"] == 1), "false_positive",
        np.where((error_rows["label"] == 1) & (error_rows["prediction"] == 0), "false_negative", "correct"),
    )
    errors = error_rows[error_rows["error_type"] != "correct"]
    errors.to_csv(output_dir / "locked_test_errors.csv", index=False)
    error_counts = {str(key): int(value) for key, value in errors["error_type"].value_counts().to_dict().items()}
    explanation_method = write_explanations(selected, x_test, y_test, output_dir)
    intervals = bootstrap_intervals(y_test, selected_scores, selected_threshold, bootstrap_iterations, SEED + 99)

    bundle = {
        "bundle_type": "prom02_raw_url_v1",
        "model": selected,
        "features": RAW_URL_FEATURES,
        "model_name": selected_name,
        "threshold": selected_threshold,
        "extractor": "url_features.extract_raw_url_features",
        "network_access": False,
        "dataset_status": "synthetic_smoke_only" if synthetic_input else "real_raw_url",
    }
    joblib.dump(bundle, output_dir / "raw_url_model.joblib")
    error_rows.to_csv(output_dir / "locked_test_predictions.csv", index=False)
    report = {
        "research_status": (
            "SMOKE TEST ONLY: this input declares itself synthetic. Metrics and model artefacts must not be used in the dissertation or deployed."
            if synthetic_input
            else "Raw-URL evaluation complete for this frozen local data snapshot; do not generalise beyond its sources/time window."
        ),
        "input": str(input_path),
        "feature_count": len(RAW_URL_FEATURES),
        "features": RAW_URL_FEATURES,
        "data_audit": audit.as_dict(),
        "outer_domain_disjoint_split": outer_manifest,
        "development_calibration_split": calibration_manifest,
        "cv_folds": n_splits,
        "selection_rule": "highest grouped-CV F1 on fit partition; test metrics were not used to select the model or threshold",
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "locked_test_metrics": metrics(y_test, selected_scores, selected_threshold),
        "locked_test_error_counts": error_counts,
        "locked_test_bootstrap_95ci": intervals,
        "explanation_method": explanation_method,
        "environment": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "joblib")
        },
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV produced by acquire_raw_url_data.py or an equivalent provenance-preserving file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="smaller tuning grids for local smoke verification only")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = train(args.input, args.output_dir, args.quick, args.bootstrap_iterations)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
