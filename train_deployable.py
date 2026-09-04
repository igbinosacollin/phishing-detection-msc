"""Historic UCI benchmark ablation, not the production raw-URL training path.

It trains on pre-extracted UCI columns; its scores therefore cannot validate the
current URL extractor or the Streamlit/email application. Use
``train_raw_url_model.py`` for the evidence-bearing shared-extractor workflow.
"""
import time, warnings, json
warnings.filterwarnings("ignore")
import pandas as pd, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from ucimlrepo import fetch_ucirepo
from url_features import DERIVABLE, LEXICAL, feature_report

SEED = 42
d = fetch_ucirepo(id=327)
X_all = d.data.features.copy()
y = (d.data.targets.iloc[:, 0] == -1).astype(int)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=SEED), {"clf__C": [0.1, 1, 10]}, True),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED), {"clf__max_depth": [6, 10, None]}, False),
    "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1), {"clf__n_estimators": [200], "clf__max_depth": [10, None]}, False),
    "SVM (RBF)": (SVC(probability=True, random_state=SEED), {"clf__C": [10], "clf__gamma": ["scale"]}, True),
    "XGBoost": (XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1), {"clf__n_estimators": [300], "clf__max_depth": [4, 6]}, False),
}

def run(cols, tag):
    X = X_all[cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows, fitted = [], {}
    for name, (est, grid, scale) in models.items():
        steps = ([("sc", StandardScaler())] if scale else []) + [("clf", est)]
        gs = GridSearchCV(Pipeline(steps), grid, cv=5, scoring="f1", n_jobs=-1)
        t0 = time.perf_counter(); gs.fit(Xtr, ytr); train_s = time.perf_counter() - t0
        m = gs.best_estimator_
        t0 = time.perf_counter(); pred = m.predict(Xte); lat = (time.perf_counter() - t0) / len(Xte) * 1000
        proba = m.predict_proba(Xte)[:, 1]
        rows.append(dict(Model=name, Accuracy=accuracy_score(yte, pred),
                         Precision=precision_score(yte, pred), Recall=recall_score(yte, pred),
                         F1=f1_score(yte, pred), ROC_AUC=roc_auc_score(yte, proba),
                         Train_time_s=round(train_s, 2), Latency_ms=round(lat, 4)))
        fitted[name] = m
    res = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    res.to_csv(f"model_comparison_{tag}.csv", index=False)
    best = res.iloc[0]["Model"]
    joblib.dump({"model": fitted[best], "features": cols, "name": best}, f"model_{tag}.joblib")
    print(f"\n--- {tag}: {len(cols)} features | best = {best} ---")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return res, best

print(feature_report())
r_full, _ = run(list(X_all.columns), "full30")
r_dep,  b_dep = run(DERIVABLE, "deployable13")
r_lex,  b_lex = run(LEXICAL, "lexical9")

gap = r_full.iloc[0]["F1"] - r_dep.iloc[0]["F1"]
json.dump({"full30_bestF1": r_full.iloc[0]["F1"], "deployable13_bestF1": r_dep.iloc[0]["F1"],
           "lexical9_bestF1": r_lex.iloc[0]["F1"], "f1_gap_full_vs_deployable": gap},
          open("deployability_gap.json", "w"), indent=2)
print(f"\nF1 gap, benchmark(30) vs deployable(13): {gap:.4f}")
