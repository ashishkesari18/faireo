"""Trust score: predict how reliable the model is in a country WITHOUT any
labels from that country, then check the prediction where labels exist.

Label-free signals per country:
  confidence   mean |p - 0.5| of the model's own predictions
  spread       std of predictions
  nbr_auc      measured AUC of the 5 countries whose landscapes look most
               similar (Presto embedding centroids)
A small linear model maps these to expected AUC. It is validated
leave-one-country-out on the countries that do have labels, then applied to
every country with at least 20 points.
  python scripts/trust.py --run lgbm_country_s0
Writes results/trust_<run>.csv
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from faireo.data import load, DATA, RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="lgbm_country_s0"); ap.add_argument("--k", type=int, default=5)
a = ap.parse_args()
X, pts = load()
p = pd.read_parquet(RESULTS / "oof" / f"{a.run}.parquet")["p"].to_numpy()
ct = pd.read_csv(RESULTS / f"countries_{a.run}.csv").set_index("country")
emb = np.load(DATA / "presto_emb.npz")["emb_noloc"]
cent = pd.DataFrame(emb).groupby(pts.country.values).mean()
cent = (cent - cent.mean()) / cent.std()

pts = pts.assign(p=p, conf=np.abs(p - 0.5))
g = pts.groupby("country").agg(n=("p", "size"), confidence=("conf", "mean"), spread=("p", "std"),
                                pred_crop_share=("p", lambda s: (s >= 0.5).mean()),
                                income=("income", "first"), iso3=("iso3", "first"),
                                lat=("lat", "median"), lon=("lon", "median"))
g = g[g.n >= 20]
labelled = list(ct.index)


def nbr_auc(c, pool):
    o = [x for x in pool if x != c]
    d = np.linalg.norm(cent.loc[o].to_numpy() - cent.loc[c].to_numpy(), axis=1)
    return ct.loc[[o[i] for i in np.argsort(d)[:a.k]], "auc"].mean()


feats = ["confidence", "spread", "nbr_auc"]
# leave-one-country-out validation on labelled countries
loo = []
for c in labelled:
    pool = [x for x in labelled if x != c]
    tr = g.loc[pool].copy(); tr["nbr_auc"] = [nbr_auc(x, [y for y in pool]) for x in pool]
    m = LinearRegression().fit(tr[feats], ct.loc[pool, "auc"])
    row = g.loc[[c]].copy(); row["nbr_auc"] = nbr_auc(c, pool)
    loo.append(m.predict(row[feats])[0])
true = ct.loc[labelled, "auc"].to_numpy(); loo = np.array(loo)
rho, pval = spearmanr(loo, true)
mae = np.abs(loo - true).mean(); base_mae = np.abs(true.mean() - true).mean()
print(f"leave-one-country-out: Spearman rho={rho:.2f} (p={pval:.4f}); MAE={mae:.3f} vs {base_mae:.3f} for a constant guess")
bottom = set(ct.loc[labelled, "auc"].nsmallest(10).index)
flag = set(np.array(labelled)[np.argsort(loo)[:10]])
print(f"of the 10 truly worst countries, {len(bottom & flag)} are among the 10 flagged as least trustworthy")

# final model on all labelled countries, applied everywhere
tr = g.loc[labelled].copy(); tr["nbr_auc"] = [nbr_auc(x, labelled) for x in labelled]
m = LinearRegression().fit(tr[feats], ct.loc[labelled, "auc"])
g["nbr_auc"] = [nbr_auc(c, labelled) for c in g.index]
g["expected_auc"] = m.predict(g[feats])
g["measured_auc"] = ct["auc"].reindex(g.index)
g.loc[labelled, "expected_auc_loo"] = loo
g.reset_index().to_csv(RESULTS / f"trust_{a.run}.csv", index=False)
print(f"trust scores for {len(g)} countries ({len(labelled)} with measured AUC)")
