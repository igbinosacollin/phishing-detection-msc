"""Does moving the decision threshold off 0.5 improve the deployable model?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, precision_recall_curve)
from ucimlrepo import fetch_ucirepo
from url_features import DERIVABLE

SEED = 42
d = fetch_ucirepo(id=327)
y = (d.data.targets.iloc[:, 0] == -1).astype(int)
X = d.data.features[DERIVABLE]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
# split a validation set OUT OF TRAINING so the threshold is not chosen on the test set
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=SEED)

model = joblib.load("model_deployable13.joblib")["model"]
model.fit(Xa, ya)
pv = model.predict_proba(Xv)[:, 1]

prec, rec, thr = precision_recall_curve(yv, pv)
f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
best_t = thr[int(np.nanargmax(f1[:-1]))]
print(f"threshold chosen on validation set: {best_t:.4f} (default 0.5)")

pt = model.predict_proba(Xte)[:, 1]
rows = []
for name, t in [("default 0.5", 0.5), (f"tuned {best_t:.3f}", best_t),
                ("recall-oriented 0.30", 0.30), ("precision-oriented 0.70", 0.70)]:
    pred = (pt >= t).astype(int)
    rows.append([name, accuracy_score(yte, pred), precision_score(yte, pred),
                 recall_score(yte, pred), f1_score(yte, pred),
                 int(((pred == 0) & (yte == 1)).sum()), int(((pred == 1) & (yte == 0)).sum())])
df = pd.DataFrame(rows, columns=["Threshold", "Accuracy", "Precision", "Recall", "F1",
                                 "Phishing missed", "False alarms"])
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
df.to_csv("threshold_comparison.csv", index=False)

# ---- figure: how the two error types trade off across the threshold ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ts = np.linspace(0.05, 0.95, 91)
missed = [int(((pt < t) & (yte == 1)).sum()) for t in ts]
alarms = [int(((pt >= t) & (yte == 0)).sum()) for t in ts]
f1s = [f1_score(yte, (pt >= t).astype(int)) for t in ts]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ts, missed, color="#C62828", lw=2, label="Phishing sites missed")
ax.plot(ts, alarms, color="#2E5E8F", lw=2, label="False alarms")
ax.axvline(0.5, color="grey", ls="--", lw=1)
ax.text(0.505, max(missed) * 0.94, "default 0.5", fontsize=8, color="grey")
ax.axvline(best_t, color="darkgreen", ls="--", lw=1)
ax.text(best_t - 0.005, max(missed) * 0.94, f"tuned {best_t:.2f}", fontsize=8,
        color="darkgreen", ha="right")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Number of test cases (n = 2,211)")
ax.set_title("Choosing an operating point: which errors the tool makes")
ax.legend(loc="upper center")
ax2 = ax.twinx(); ax2.plot(ts, f1s, color="#C1602F", lw=1.2, alpha=.7)
ax2.set_ylabel("F1", color="#C1602F"); ax2.tick_params(axis="y", labelcolor="#C1602F")
plt.tight_layout(); plt.savefig("plot_threshold.png", dpi=150); plt.close()
print("wrote plot_threshold.png")
