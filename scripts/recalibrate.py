"""Post-hoc fixes on out-of-fold predictions (countries held out).

  python scripts/recalibrate.py --run lgbm_country_s0

Methods compared for every country the model never saw:
  none            the model as trained
  global_platt    recalibrated on all other unseen-country predictions
  nbr_feature     recalibrated on the 8 countries that look most similar (Presto space)  <- spatially aware, no local labels
  nbr_geo         recalibrated on the 8 geographically closest countries               <- spatially aware, no local labels
  em_prior        prevalence re-estimated from unlabelled predictions (Saerens EM)
  local_k         recalibrated with k labelled points from the country (k=20,50,100)
Writes results/fix_<run>.csv (one row per country x method).
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from faireo.data import load, eligible_countries, DATA, RESULTS
from faireo.metrics import point_metrics
from faireo.fixes import fit_platt, apply_platt, neighbour_recalibrate, em_prior_shift, local_recalibrate

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="lgbm_country_s0")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--repeats", type=int, default=20)
a = ap.parse_args()

X, pts = load()
p = pd.read_parquet(RESULTS / "oof" / f"{a.run}.parquet")["p"].to_numpy()
y = pts.y.to_numpy()
countries = eligible_countries(pts)
emb = np.load(DATA / "presto_emb.npz")["emb_noloc"]
cent_feat = pd.DataFrame(emb).groupby(pts.country.values).mean()
cent_feat = (cent_feat - cent_feat.mean()) / cent_feat.std()
xyz = np.stack([np.cos(np.radians(pts.lat)) * np.cos(np.radians(pts.lon)),
                np.cos(np.radians(pts.lat)) * np.sin(np.radians(pts.lon)),
                np.sin(np.radians(pts.lat))], 1)
cent_geo = pd.DataFrame(xyz).groupby(pts.country.values).mean()

rows = []
rng = np.random.default_rng(0)
for c in countries:
    t = (pts.country == c).to_numpy()
    yt, pt = y[t], p[t]
    info = dict(country=c, income=pts.loc[t, "income"].iloc[0], true_share=yt.mean())

    def add(method, q, yy=yt, share=None):
        m = point_metrics(yy, q)
        est = q.mean() if share is None else share
        rows.append({**info, "method": method, **{k: m[k] for k in ["auc", "bal_acc", "f1", "ece", "brier"]},
                     "area_err_pp": m["area_err_pp"], "share_err_pp": 100 * (est - yy.mean())})

    add("none", pt)
    o = (~t) & pts.country.isin(countries).to_numpy()
    add("global_platt", apply_platt(pt, fit_platt(p[o], y[o])))
    add("nbr_feature", neighbour_recalibrate(pts, p, c, cent_feat, a.k, countries)[0])
    add("nbr_geo", neighbour_recalibrate(pts, p, c, cent_geo, a.k, countries)[0])
    q, pi = em_prior_shift(pt, y[~t].mean())
    add("em_prior", q, share=pi)
    for k in (20, 50, 100):
        if len(yt) - k < 50:
            continue
        acc = []
        for r in range(a.repeats):
            qq, rest = local_recalibrate(pt, yt, k, rng)
            m0, m1 = point_metrics(yt[rest], pt[rest]), point_metrics(yt[rest], qq)
            acc.append({key: m1[key] for key in ["auc", "bal_acc", "f1", "ece", "brier", "area_err_pp"]})
        mean = pd.DataFrame(acc).mean().to_dict()
        rows.append({**info, "method": f"local_{k}", **mean, "share_err_pp": mean["area_err_pp"]})

R = pd.DataFrame(rows)
R.to_csv(RESULTS / f"fix_{a.run}.csv", index=False)
R["abs_area_err"] = R.area_err_pp.abs(); R["abs_share_err"] = R.share_err_pp.abs()
summ = R.groupby("method").agg(bal_acc=("bal_acc", "mean"), worst_bal_acc=("bal_acc", "min"),
                               ece=("ece", "mean"), abs_area_err=("abs_area_err", "mean"),
                               abs_share_err=("abs_share_err", "mean"))
low = R[R.income == "Low"].groupby("method")[["bal_acc", "abs_share_err"]].mean().add_prefix("low_")
high = R[R.income == "High"].groupby("method")[["bal_acc"]].mean().add_prefix("high_")
S = summ.join(low).join(high)
S["gap_bal_acc"] = S.high_bal_acc - S.low_bal_acc
print(S.round(3).to_string())
