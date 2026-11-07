"""Audit a map people already use: ESA WorldCover cropland vs ground labels.

Prediction = WorldCover class 40 (cropland) at the point; label = CropHarvest.
Uses the WorldCover year closest to the label year (2020 for <=2020, else 2021).
  python scripts/worldcover_audit.py
Writes results/countries_worldcover.csv and prints the by-income summary.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from faireo.data import load, eligible_countries, DATA, RESULTS
from faireo.metrics import country_table, gap_ci

X, pts = load()
wc = pd.read_parquet(DATA / "worldcover.parquet")
cls = np.where(pts.year.fillna(2020).to_numpy() <= 2020, wc.wc2020, wc.wc2021)
ok = cls > 0
pred = (cls == 40).astype(float)
print(f"WorldCover read at {ok.mean():.1%} of points")
sub = pts[ok].reset_index(drop=True); p = pred[ok]
cs = [c for c in eligible_countries(pts) if (sub.country == c).sum() >= 100]
ct = country_table(sub, p, cs, keys=("bal_acc", "f1"))
ct.to_csv(RESULTS / "countries_worldcover.csv", index=False)
for m in ["bal_acc", "recall", "specificity", "area_err_pp"]:
    print(f"{m:12s}", ct.groupby("income")[m].mean().round(3).to_dict())
print("bal_acc gap high-low (95% CI):", np.round(gap_ci(ct, "bal_acc"), 3))
print("recall gap high-low (95% CI):", np.round(gap_ci(ct, "recall"), 3))
print("cropland share error, pp (map minus truth), by income:",
      ct.groupby("income").area_err_pp.mean().round(1).to_dict())
gw = sub.dataset.eq("geowiki-landcover-2017").to_numpy()
s2 = sub[gw].reset_index(drop=True)
cs2 = [c for c in cs if (s2.country == c).sum() >= 100 and 20 <= s2[s2.country == c].y.sum() <= (s2.country == c).sum() - 20]
ct2 = country_table(s2, p[gw], cs2, keys=("bal_acc",))
print(f"Geo-Wiki labels only ({len(cs2)} countries): recall", ct2.groupby("income").recall.mean().round(3).to_dict(),
      "bal_acc gap CI", np.round(gap_ci(ct2, "bal_acc"), 3))
