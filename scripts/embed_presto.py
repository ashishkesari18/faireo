"""Compute Presto embeddings once and cache them in data/presto_emb.npz."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from faireo.data import load, DATA
from faireo.presto_embed import embed

X, pts = load()
t = time.time()
e = embed(X, pts.lat.values, pts.lon.values, use_latlon=True)
e0 = embed(X, pts.lat.values, pts.lon.values, use_latlon=False)
np.savez_compressed(DATA / "presto_emb.npz", emb=e, emb_noloc=e0)
print(f"embedded {len(e):,} points in {time.time()-t:.0f}s -> data/presto_emb.npz")
