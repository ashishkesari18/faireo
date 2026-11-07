"""Compute Prithvi-EO-2.0-tiny embeddings once; cache in data/prithvi_emb.npz."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from faireo.data import load, DATA
from faireo.prithvi_embed import embed

X, pts = load()
t = time.time()
year = pts.year.fillna(pts.year.median()).to_numpy()
e = embed(X, pts.lat.values, pts.lon.values, year)
np.savez_compressed(DATA / "prithvi_emb.npz", emb=e)
print(f"embedded {len(e):,} points in {time.time()-t:.0f}s -> data/prithvi_emb.npz")
