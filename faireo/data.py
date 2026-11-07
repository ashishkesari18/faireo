"""Load the prepared CropHarvest table and build model inputs."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

BANDS = ["VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9",
         "B11", "B12", "temperature_2m", "total_precipitation", "elevation",
         "slope", "NDVI"]
INCOME_LEVELS = ["Low", "Lower-middle", "Upper-middle", "High"]

# a country is reported on only if it has enough points of both classes
MIN_POINTS = 100
MIN_PER_CLASS = 20


def load():
    """Returns X [N, 12, 18] raw values, and a points table (one row per point)."""
    d = np.load(DATA / "cropharvest.npz")
    pts = pd.read_parquet(DATA / "points.parquet").reset_index(drop=True)
    return d["X"], pts


def tabular(X):
    """Features for tree models: all 12x18 monthly values plus annual summaries."""
    n = len(X)
    flat = X.reshape(n, -1)
    stats = np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)
    ndvi = X[:, :, BANDS.index("NDVI")]
    extra = np.stack([ndvi.max(1) - ndvi.min(1), ndvi.argmax(1)], axis=1)
    return np.concatenate([flat, stats, extra], axis=1).astype(np.float32)


def eligible_countries(pts):
    g = pts.groupby("country")["y"].agg(["size", "sum"])
    ok = (g["size"] >= MIN_POINTS) & (g["sum"] >= MIN_PER_CLASS) & (g["size"] - g["sum"] >= MIN_PER_CLASS)
    return sorted(g.index[ok])
