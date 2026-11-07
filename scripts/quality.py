"""Data-quality table: before blaming the model, check what the data look like.

Per country: points, crop share, number of label sources, share of labels from
the global crowd-sourced Geo-Wiki campaign, years, and how often the Sentinel-2
series is missing (all-zero months, e.g. persistent cloud).
Writes results/data_quality.csv and prints a by-income summary.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from faireo.data import load, eligible_countries, BANDS, RESULTS

X, pts = load()
s2 = [BANDS.index(b) for b in ["B2", "B3", "B4", "B8"]]
pts["s2_missing_months"] = (np.abs(X[:, :, s2]).sum(2) == 0).sum(1)
pts["geowiki"] = pts.dataset.eq("geowiki-landcover-2017")
c = eligible_countries(pts)
q = (pts[pts.country.isin(c)].groupby(["country", "income", "continent"])
     .agg(points=("y", "size"), crop_share=("y", "mean"), sources=("dataset", "nunique"),
          geowiki_share=("geowiki", "mean"), year_min=("year", "min"), year_max=("year", "max"),
          s2_missing_months=("s2_missing_months", "mean")).reset_index())
q.to_csv(RESULTS / "data_quality.csv", index=False)
print(q.groupby("income")[["points", "crop_share", "sources", "geowiki_share", "s2_missing_months"]]
      .median().round(2).to_string())
