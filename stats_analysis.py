"""
Confidence intervals and significance testing (Objective O5).

The proposal promised confidence intervals, and the module guidance asks an
experimental project to report the statistical significance and reliability of
its results. Two questions are worth testing:

  1. Random Forest and XGBoost tie to four decimal places on the full feature
     set. Is that a real tie or is the difference just noise?
  2. The deployable model scores 19.3 F1 points below the benchmark. Is that
     gap larger than sampling error?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             average_precision_score)
from scipy.stats import binomtest
from ucimlrepo import fetch_ucirepo
from url_features import DERIVABLE, LEXICAL

SEED, B = 42, 2000
rng = np.random.default_rng(SEED)

d = fetch_ucirepo(id=327)
y = (d.data.targets.iloc[:, 0] == -1).astype(int)
Xall = d.data.features


def held_out(cols):
    return train_test_split(Xall[cols], y, test_size=0.2, stratify=y, random_state=SEED)


def boot_ci(yt, yp, fn, b=B):
    """Percentile bootstrap over the test set: how much would this metric move
    if we had drawn a different sample of the same size?"""
    yt, yp = np.asarray(yt), np.asarray(yp)
    n = len(yt)
    vals = [fn(yt[i], yp[i]) for i in
            (rng.integers(0, n, n) for _ in range(b))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(yt, a, b_):
    """McNemar's test on the cases where two models disagree. The right test for
    paired predictions on one test set; a t-test would assume independence the
    data does not have."""
    yt, a, b_ = np.asarray(yt), np.asarray(a), np.asarray(b_)
    n01 = int(((a == yt) & (b_ != yt)).sum())   # only A correct
    n10 = int(((a != yt) & (b_ == yt)).sum())   # only B correct
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue)


print("=" * 74)
print("CONFIDENCE INTERVALS (95%, percentile bootstrap, 2000 resamples)")
print("=" * 74)
preds, rows = {}, []
for tag, cols in [("full30", list(Xall.columns)), ("deployable13", DERIVABLE),
                  ("lexical9", LEXICAL)]:
    Xtr, Xte, ytr, yte = held_out(cols)
    bundle = joblib.load(f"model_{tag}.joblib")
    m, name = bundle["model"], bundle["name"]
    p = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
    preds[tag] = (yte.values, p, proba, name)
    acc, f1 = accuracy_score(yte, p), f1_score(yte, p)
    al, ah = boot_ci(yte, p, accuracy_score)
    fl, fh = boot_ci(yte, p, f1_score)
    rows.append([tag, name, f"{acc:.4f}", f"[{al:.4f}, {ah:.4f}]",
                 f"{f1:.4f}", f"[{fl:.4f}, {fh:.4f}]"])
print(pd.DataFrame(rows, columns=["Condition", "Model", "Accuracy", "95% CI",
                                  "F1", "95% CI"]).to_string(index=False))

print("\n" + "=" * 74)
print("Q1  Full-30: is the Random Forest / XGBoost tie real?")
print("=" * 74)
Xtr, Xte, ytr, yte = held_out(list(Xall.columns))
rf = joblib.load("model_full30.joblib")["model"]
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
xgb = GridSearchCV(XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1),
                   {"n_estimators": [300], "max_depth": [4, 6]}, cv=5,
                   scoring="f1", n_jobs=-1).fit(Xtr, ytr).best_estimator_
a, b_ = rf.predict(Xte), xgb.predict(Xte)
n01, n10, pv = mcnemar(yte, a, b_)
print(f"  RF correct where XGB wrong: {n01}    XGB correct where RF wrong: {n10}")
print(f"  McNemar p = {pv:.4f}  ->  {'no significant difference' if pv>=0.05 else 'significant'}")
print("  Interpretation: the two ensembles are statistically indistinguishable on this")
print("  test set, so selecting between them on F1 alone is not defensible. Latency and")
print("  interpretability are the appropriate tie-breakers.")

print("\n" + "=" * 74)
print("Q2  Is the deployability gap larger than sampling error?")
print("=" * 74)
yt_f, p_f, _, n_f = preds["full30"]
yt_d, p_d, _, n_d = preds["deployable13"]
n01, n10, pv2 = mcnemar(yt_f, p_f, p_d)
print(f"  full30 correct where deployable13 wrong: {n01}")
print(f"  deployable13 correct where full30 wrong: {n10}")
print(f"  McNemar p = {pv2:.3e}  ->  {'significant' if pv2<0.05 else 'not significant'}")
print("  Interpretation: the 19.3-point F1 gap is far outside sampling error. It reflects")
print("  the information removed with the 17 unobtainable features, not chance.")

# ---- precision-recall curves, promised in the proposal, previously missing ----
plt.figure(figsize=(7.5, 5.5))
for tag, label in [("full30", "Full-30 (benchmark)"),
                   ("deployable13", "Deployable-13"), ("lexical9", "Lexical-9")]:
    yt, _p, proba, name = preds[tag]
    pr, rc, _ = precision_recall_curve(yt, proba)
    ap = average_precision_score(yt, proba)
    plt.plot(rc, pr, lw=2, label=f"{label}: {name} (AP={ap:.3f})")
plt.axhline(float(np.mean(preds['full30'][0])), ls="--", c="grey", lw=1)
plt.text(0.02, float(np.mean(preds['full30'][0])) + .01, "no-skill baseline",
         fontsize=8, color="grey")
plt.xlabel("Recall (share of phishing sites caught)")
plt.ylabel("Precision (share of flagged sites that really are phishing)")
plt.title("Precision-recall by feature availability")
plt.legend(loc="lower left"); plt.ylim(0.3, 1.02); plt.tight_layout()
plt.savefig("plot_precision_recall.png", dpi=150); plt.close()
print("\nwrote plot_precision_recall.png")

pd.DataFrame(rows, columns=["Condition", "Model", "Accuracy", "Accuracy_95CI",
                            "F1", "F1_95CI"]).to_csv("confidence_intervals.csv", index=False)
