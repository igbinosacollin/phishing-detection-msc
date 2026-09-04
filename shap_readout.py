"""Numbers behind Figure 4.6 so section 4.4 can be written from evidence."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, shap
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from url_features import DERIVABLE

SEED = 42
d = fetch_ucirepo(id=327)
y = (d.data.targets.iloc[:, 0] == -1).astype(int)
X = d.data.features[DERIVABLE]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

dep = joblib.load("model_deployable13.joblib")
model = dep["model"]
clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
sample = Xte.sample(400, random_state=SEED)
sv = shap.TreeExplainer(clf).shap_values(sample)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)

print("=== GLOBAL: mean |SHAP|, i.e. how much each feature moves predictions ===")
imp = pd.Series(np.abs(sv).mean(0), index=sample.columns).sort_values(ascending=False)
tot = imp.sum()
for f, v in imp.items():
    print(f"  {f:<30} {v:.4f}   {100*v/tot:5.1f}% of total")

print("\n=== DIRECTION: mean SHAP when feature = -1 (phishy) vs +1 (legit) ===")
print("  positive SHAP pushes the prediction toward PHISHING")
for f in imp.index[:6]:
    row = f"  {f:<30}"
    for val in (-1, 0, 1):
        m = sample[f] == val
        row += f" [{val:+d}]={sv[m.values, sample.columns.get_loc(f)].mean():+.3f} " if m.sum() else f" [{val:+d}]=n/a "
    print(row + f" (n={len(sample)})")

print("\n=== WORKED EXAMPLE: one test URL the model flags as phishing ===")
proba = model.predict_proba(sample)[:, 1]
k = int(np.argmax(proba))
inst = sample.iloc[k]
print(f"  true label: {'phishing' if yte.loc[sample.index[k]]==1 else 'legitimate'}"
      f"   predicted P(phishing) = {proba[k]:.4f}")
contrib = pd.Series(sv[k], index=sample.columns).sort_values(key=abs, ascending=False)
print(f"  {'feature':<30}{'value':>7}{'SHAP':>10}  effect")
for f, s in contrib.head(6).items():
    print(f"  {f:<30}{int(inst[f]):>7}{s:>10.4f}  {'toward phishing' if s>0 else 'toward legitimate'}")

print("\n=== A legitimate example for contrast ===")
k2 = int(np.argmin(proba))
inst2 = sample.iloc[k2]
print(f"  true label: {'phishing' if yte.loc[sample.index[k2]]==1 else 'legitimate'}"
      f"   predicted P(phishing) = {proba[k2]:.4f}")
c2 = pd.Series(sv[k2], index=sample.columns).sort_values(key=abs, ascending=False)
for f, s in c2.head(4).items():
    print(f"  {f:<30}{int(inst2[f]):>7}{s:>10.4f}  {'toward phishing' if s>0 else 'toward legitimate'}")
