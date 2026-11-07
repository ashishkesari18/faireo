"""Is the gap caused by different labelling campaigns?

Geo-Wiki 2017 labelled points all over the world with one protocol. Scoring
only Geo-Wiki points removes differences in label source between countries.
  python scripts/label_control.py --run lgbm_country_s0
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from faireo.data import load, RESULTS
from faireo.metrics import country_table, gap_ci, group_summary

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="lgbm_country_s0"); a = ap.parse_args()
X, pts = load()
p = pd.read_parquet(RESULTS / "oof" / f"{a.run}.parquet")["p"].to_numpy()
gw = pts.dataset.eq("geowiki-landcover-2017").to_numpy()
sub = pts[gw].reset_index(drop=True); ps = p[gw]
g = sub.groupby("country")["y"].agg(["size", "sum"])
cs = sorted(g.index[(g["size"] >= 100) & (g["sum"] >= 20) & (g["size"] - g["sum"] >= 20)])
ct = country_table(sub, ps, cs)
ct.to_csv(RESULTS / f"countries_geowiki_{a.run}.csv", index=False)
print(f"Geo-Wiki points only, {len(cs)} countries")
for m in ["auc", "recall", "pred_prevalence", "prevalence"]:
    print(m, ct.groupby("income")[m].mean().round(3).to_dict())
print("AUC gap high-low (95% CI):", np.round(gap_ci(ct, "auc"), 3))
print("Recall gap high-low (95% CI):", np.round(gap_ci(ct, "recall"), 3))
