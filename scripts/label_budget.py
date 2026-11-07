"""Where should the next labels be collected?

Step 1 (learning curves): for each country held out, add k of its own
labelled points to training (k = 0, 25, 100, 400) and score the rest.
Model: logistic regression on Presto embeddings (fast and strong).
Step 2 (allocation): split a fixed budget across countries with different
policies and read the expected accuracy off the curves:
  status_quo   proportional to the labels a country already has
  uniform      the same number for every country
  novelty      proportional to how unfamiliar the country looks to the model
               (label-free: computable before collecting anything)
  oracle       greedy on the true curves (upper bound, not usable in practice)
  python scripts/label_budget.py
Writes results/label_curves.csv and results/label_budget.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from faireo.data import load, eligible_countries, DATA, RESULTS

KS = [0, 25, 100, 400]
REPEATS = 3
LOCAL_WEIGHT = 5.0   # local labels count 5x in training

X, pts = load()
E = StandardScaler().fit_transform(np.load(DATA / "presto_emb.npz")["emb"])
y = pts.y.to_numpy()
countries = eligible_countries(pts)
rng = np.random.default_rng(0)
curves_f = RESULTS / "label_curves.csv"
if curves_f.exists():
    curves = pd.read_csv(curves_f)
else:
    rows = []
    # subsample the training pool for speed; the held-out country is always complete
    for c in countries:
        t = np.where(pts.country.to_numpy() == c)[0]
        o = np.where(pts.country.to_numpy() != c)[0]
        o = rng.choice(o, min(30000, len(o)), replace=False)
        for k in KS:
            if k >= len(t) - 50:
                continue
            for r in range(REPEATS if k else 1):
                perm = rng.permutation(t); lab, test = perm[:k], perm[k:]
                tr = np.concatenate([o, lab])
                w = np.concatenate([np.ones(len(o)), np.full(len(lab), LOCAL_WEIGHT)])
                m = LogisticRegression(C=1.0, max_iter=1000).fit(E[tr], y[tr], sample_weight=w)
                q = m.predict_proba(E[test])[:, 1]
                rows.append(dict(country=c, k=k, rep=r, auc=roc_auc_score(y[test], q)))
        print(c, flush=True)
    curves = pd.DataFrame(rows).groupby(["country", "k"]).auc.mean().reset_index()
    curves.to_csv(curves_f, index=False)

# ---- allocation ----
info = pts[pts.country.isin(countries)].groupby("country").agg(n=("y", "size"), income=("income", "first"))
nov = pd.read_csv(RESULTS / "novelty.csv").set_index("country")["novelty"]
C = curves.pivot(index="country", columns="k", values="auc")


def auc_at(c, k):
    """Interpolate the learning curve in log(1+k)."""
    row = C.loc[c].dropna()
    ks = np.log1p(row.index.to_numpy(float)); v = row.to_numpy()
    return float(np.interp(np.log1p(k), ks, v))


def evaluate(alloc):
    a = pd.Series({c: auc_at(c, alloc.get(c, 0)) for c in C.index})
    low = a[info.loc[a.index, "income"] == "Low"]
    return dict(mean_auc=a.mean(), worst10_auc=a.nsmallest(max(1, len(a) // 10)).mean(),
                low_income_auc=low.mean(), min_auc=a.min())


def proportional(weights, B):
    w = weights / weights.sum()
    return {c: B * w[c] for c in w.index}


rows = []
for B in [0, 1000, 2500, 5000, 10000]:
    pol = {
        "status_quo": proportional(info.loc[C.index, "n"].astype(float), B),
        "uniform": {c: B / len(C) for c in C.index},
        "novelty": proportional(nov.loc[C.index] ** 4, B),
    }
    # oracle greedy in chunks of 25 labels
    alloc = {c: 0.0 for c in C.index}
    for _ in range(int(B // 25)):
        gain = {c: auc_at(c, alloc[c] + 25) - auc_at(c, alloc[c]) + 1e-6 * (1 - auc_at(c, alloc[c])) for c in C.index}
        # focus on the worst-off: weight gains by how bad the country currently is
        best = max(C.index, key=lambda c: gain[c] * (1 - auc_at(c, alloc[c])))
        alloc[best] += 25
    pol["oracle_worst_first"] = alloc
    for name, al in pol.items():
        rows.append(dict(budget=B, policy=name, **evaluate(al)))
R = pd.DataFrame(rows)
R.to_csv(RESULTS / "label_budget.csv", index=False)
print(R.round(3).to_string(index=False))
