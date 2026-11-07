"""Out-of-fold predictions for every point.

  python scripts/audit.py --model lgbm --split country   # whole countries held out
  python scripts/audit.py --model lgbm --split random    # countries seen in training

Writes results/oof/<model>_<split>_s<seed>.parquet (columns: p, fold).
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from faireo.data import load, RESULTS
from faireo.models import MODELS
from faireo.splits import SPLITS
from faireo.fixes import balanced_weights

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--split", default="country", choices=list(SPLITS))
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--balance", type=float, default=0.0,
                help="0 = ordinary training; 0.5 or 1 = give small countries more weight (the fix)")
a = ap.parse_args()

X, pts = load()
feats, make = MODELS[a.model]
F = feats(X, pts)
y = pts.y.to_numpy()
p = np.full(len(y), np.nan); fold = np.full(len(y), -1)
t = time.time()
for k, (tr, te) in enumerate(SPLITS[a.split](pts, a.folds, a.seed)):
    m = make()
    if a.balance > 0:
        w = balanced_weights(pts, tr, a.balance)
        if hasattr(m, "steps"):
            m.fit(F[tr], y[tr], **{f"{m.steps[-1][0]}__sample_weight": w})
        else:
            m.fit(F[tr], y[tr], sample_weight=w)
    else:
        m.fit(F[tr], y[tr])
    p[te] = m.predict_proba(F[te])[:, 1]; fold[te] = k
    print(f"  fold {k+1}/{a.folds} done ({time.time()-t:.0f}s)", flush=True)
out = RESULTS / "oof"; out.mkdir(parents=True, exist_ok=True)
tag = f"_bal{a.balance:g}" if a.balance > 0 else ""
f = out / f"{a.model}{tag}_{a.split}_s{a.seed}.parquet"
pd.DataFrame({"p": p, "fold": fold}).to_parquet(f)
print(f"saved {f.relative_to(RESULTS.parent)}")
