"""Per-country metrics and income-group summaries for every saved prediction file.

  python scripts/report.py            # all files in results/oof/
Writes results/countries_<name>.csv and results/summary.csv, and prints the headline table.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from faireo.data import load, eligible_countries, RESULTS
from faireo.metrics import country_table, group_summary, gap_ci

X, pts = load()
countries = eligible_countries(pts)
rows = []
for f in sorted((RESULTS / "oof").glob("*.parquet")):
    name = f.stem
    p = pd.read_parquet(f)["p"].to_numpy()
    ct = country_table(pts, p, countries)
    ct.to_csv(RESULTS / f"countries_{name}.csv", index=False)
    for metric in ["auc", "bal_acc", "ece"]:
        gs = group_summary(ct, metric)
        g, lo, hi = gap_ci(ct, metric)
        r = dict(run=name, metric=metric, countries=len(ct), mean=ct[metric].mean(),
                 worst=ct[metric].min() if metric != "ece" else ct[metric].max(),
                 gap_high_minus_low=g, gap_lo=lo, gap_hi=hi)
        for _, s in gs.iterrows():
            r[f"{s['income']}"] = s["mean"]
        rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(RESULTS / "summary.csv", index=False)
pd.set_option("display.width", 200)
print(f"{len(countries)} countries with >= 100 points and both classes\n")
print(S.round(3).to_string(index=False))
