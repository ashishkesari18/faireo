"""Why are some countries worse? Regress per-country AUC on candidate causes.

Candidates: income group, how unfamiliar the country looks to the model
(novelty, label-free), distance to the nearest labelled point in another
country, crop share, share of Geo-Wiki labels, and number of points.
  python scripts/explain.py --run lgbm_country_s0
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from faireo.data import RESULTS

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="lgbm_country_s0"); a = ap.parse_args()
ct = pd.read_csv(RESULTS / f"countries_{a.run}.csv")
seen = pd.read_csv(RESULTS / f"countries_{a.run.replace('_country_', '_random_')}.csv")[["country", "auc"]]
q = pd.read_csv(RESULTS / "data_quality.csv")[["country", "geowiki_share", "points"]]
nov = pd.read_csv(RESULTS / "novelty.csv")[["country", "novelty", "nearest_other_km"]]
d = ct.merge(seen, on="country", suffixes=("", "_seen")).merge(q, on="country").merge(nov, on="country")
d["drop"] = d.auc_seen - d.auc
d["low_income"] = d.income.isin(["Low"]).astype(int)
d["log_km"] = np.log10(d.nearest_other_km + 1); d["log_n"] = np.log10(d.points)
z = lambda s: (s - s.mean()) / s.std()
for c in ["novelty", "log_km", "prevalence", "geowiki_share", "log_n"]:
    d[c + "_z"] = z(d[c])
d.to_csv(RESULTS / f"explain_{a.run}.csv", index=False)

print("Spearman correlation with AUC on unseen countries (n=%d):" % len(d))
for c in ["novelty", "nearest_other_km", "prevalence", "geowiki_share", "points"]:
    r, p = spearmanr(d[c], d.auc); print(f"  {c:18s} rho={r:+.2f}  p={p:.3f}")
r, p = spearmanr(d.novelty, d["drop"]); print(f"  novelty vs drop from being unseen: rho={r:+.2f} p={p:.3f}")
for f in ["auc ~ low_income", "auc ~ low_income + novelty_z",
          "auc ~ low_income + novelty_z + log_km_z + prevalence_z + geowiki_share_z + log_n_z"]:
    m = smf.ols(f, d).fit(cov_type="HC3")
    print(f"\n{f}   (R2={m.rsquared:.2f})")
    print(pd.DataFrame({"coef": m.params, "lo": m.conf_int()[0], "hi": m.conf_int()[1], "p": m.pvalues}).round(3).to_string())
