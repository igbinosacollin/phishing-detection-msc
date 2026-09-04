"""Regenerate every dissertation figure from the REAL UCI data."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, joblib, shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo
from url_features import DERIVABLE

SEED = 42
d = fetch_ucirepo(id=327)
X_all = d.data.features.copy(); y = (d.data.targets.iloc[:, 0] == -1).astype(int)

def split(cols):
    return train_test_split(X_all[cols], y, test_size=0.2, stratify=y, random_state=SEED)

# ---- Fig 1: ROC curves, all five models, full-30 -----------------------------
_, Xte, _, yte = split(list(X_all.columns))
import itertools
from sklearn.model_selection import GridSearchCV  # noqa (models already tuned; reload)
full = joblib.load("model_full30.joblib")
plt.figure(figsize=(7, 6))
# reuse the saved best model + retrain-free ROC for the rest via the comparison csv
res = pd.read_csv("model_comparison_full30.csv")
fpr, tpr, _ = roc_curve(yte, full["model"].predict_proba(Xte)[:, 1])
plt.plot(fpr, tpr, lw=2, label=f"{full['name']} (AUC={auc(fpr,tpr):.4f})")
plt.plot([0, 1], [0, 1], "k--", alpha=.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC — best model, full 30-feature set (real UCI data)")
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig("plot_roc.png", dpi=150); plt.close()

# ---- Fig 2: metric comparison across the five models -------------------------
ax = res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].plot(
    kind="bar", figsize=(9, 5))
ax.set_ylim(0.85, 1.0); ax.set_ylabel("Score")
ax.set_title("Model performance — full 30-feature set (real UCI data)")
plt.xticks(rotation=20, ha="right"); plt.legend(fontsize=8, ncol=5)
plt.tight_layout(); plt.savefig("plot_metrics.png", dpi=150); plt.close()

# ---- Fig 3: confusion matrix, best full-30 model -----------------------------
cm = confusion_matrix(yte, full["model"].predict(Xte))
ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phish"]).plot(cmap="Blues", values_format="d")
plt.title(f"Confusion matrix — {full['name']} (full 30)")
plt.tight_layout(); plt.savefig("plot_confusion.png", dpi=150); plt.close()

# ---- Fig 4: THE key figure — three-tier deployability gap --------------------
tiers = [("Full-30\n(benchmark)", "full30"), ("Deployable-13\n(URL + WHOIS)", "deployable13"),
         ("Lexical-9\n(URL only)", "lexical9")]
f1s, accs, names = [], [], []
for label, tag in tiers:
    t = pd.read_csv(f"model_comparison_{tag}.csv").iloc[0]
    f1s.append(t["F1"]); accs.append(t["Accuracy"]); names.append(f"{label}\nbest: {t['Model']}")
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, accs, w, label="Accuracy", color="#2E5E8F")
ax.bar(x + w/2, f1s, w, label="F1", color="#C1602F")
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax.text(i - w/2, a + .008, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, f + .008, f"{f:.3f}", ha="center", fontsize=9)
ax.annotate("", xy=(0, f1s[0]), xytext=(1, f1s[1]),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.6))
ax.text(0.5, (f1s[0]+f1s[1])/2 + .02, f"ΔF1 = {f1s[0]-f1s[1]:.4f}",
        ha="center", color="crimson", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0.6, 1.03)
ax.set_ylabel("Score"); ax.set_title("Deployability gap: performance by obtainable feature set")
ax.legend(); plt.tight_layout(); plt.savefig("plot_deployability_gap.png", dpi=150); plt.close()

# ---- Fig 5: feature importance, marked by obtainability ----------------------
rf = full["model"].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=full["features"]).sort_values(ascending=False).head(12)
colors = ["#2E7D32" if f in DERIVABLE else "#C62828" for f in imp.index]
plt.figure(figsize=(8, 5.5))
plt.barh(range(len(imp))[::-1], imp.values, color=colors)
plt.yticks(range(len(imp))[::-1], imp.index, fontsize=9)
plt.xlabel("Gini importance")
plt.title("Feature importance — green = obtainable from URL, red = requires visiting page")
plt.tight_layout(); plt.savefig("plot_importance.png", dpi=150); plt.close()

# ---- Fig 6: SHAP for the DEPLOYABLE model (what the app actually uses) -------
dep = joblib.load("model_deployable13.joblib")
_, Xte13, _, _ = split(DERIVABLE)
clf = dep["model"].named_steps["clf"] if hasattr(dep["model"], "named_steps") else dep["model"]
sv = shap.TreeExplainer(clf).shap_values(Xte13.sample(min(400, len(Xte13)), random_state=SEED))
svp = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)
shap.summary_plot(svp, Xte13.sample(min(400, len(Xte13)), random_state=SEED), max_display=13, show=False)
plt.title(f"SHAP — {dep['name']}, deployable 13-feature model", fontsize=10)
plt.tight_layout(); plt.savefig("plot_shap_deployable.png", dpi=150); plt.close()

print("regenerated:", ", ".join(["plot_roc", "plot_metrics", "plot_confusion",
      "plot_deployability_gap", "plot_importance", "plot_shap_deployable"]))
