"""How unfamiliar does each country look to a model trained on the others?

For every point: distance (in Presto embedding space) to its 10 nearest
training points from *other* countries, and geographic distance (km) to the
nearest labelled point in another country. Averaged per country these are
the 'novelty' scores used by the trust map and the gap explanation.
Writes results/novelty.csv (per country) and data/novelty_points.parquet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.preprocessing import StandardScaler
from faireo.data import load, eligible_countries, DATA, RESULTS

X, pts = load()
emb = StandardScaler().fit_transform(np.load(DATA / "presto_emb.npz")["emb_noloc"])
rng = np.random.default_rng(0)
feat_d = np.full(len(pts), np.nan); geo_km = np.full(len(pts), np.nan)
rad = np.radians(pts[["lat", "lon"]].to_numpy())
for c, g in pts.groupby("country"):
    i = g.index.to_numpy()
    other = np.where(pts.country.to_numpy() != c)[0]
    ref = rng.choice(other, min(30000, len(other)), replace=False)
    nn = NearestNeighbors(n_neighbors=10).fit(emb[ref])
    feat_d[i] = nn.kneighbors(emb[i])[0].mean(1)
    geo_km[i] = BallTree(rad[other], metric="haversine").query(rad[i], k=1)[0][:, 0] * 6371
pts["novelty"] = feat_d; pts["nearest_other_km"] = geo_km
pts[["novelty", "nearest_other_km"]].to_parquet(DATA / "novelty_points.parquet")
nov = pts.groupby("country")[["novelty", "nearest_other_km"]].median().reset_index()
nov["eligible"] = nov.country.isin(eligible_countries(pts))
nov.to_csv(RESULTS / "novelty.csv", index=False)
print(nov[nov.eligible].describe().round(2))
