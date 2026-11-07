"""Per-country metrics with bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

INCOME_LEVELS = ["Low", "Lower-middle", "Upper-middle", "High"]


def ece(y, p, bins=10):
    """Expected calibration error: average gap between confidence and frequency."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def point_metrics(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p)
    pred = p >= thr
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    out = dict(n=len(y), prevalence=y.mean(), pred_prevalence=pred.mean(),
               mean_prob=p.mean(), precision=prec, recall=rec, specificity=spec,
               bal_acc=(rec + spec) / 2, f1=2 * prec * rec / max(prec + rec, 1e-9),
               brier=np.mean((p - y) ** 2), ece=ece(y, p))
    out["auc"] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    # cropland share error: what an area estimate would get wrong (percentage points)
    out["area_err_pp"] = 100 * (pred.mean() - y.mean())
    return out


def country_table(pts, p, countries, B=200, seed=0, keys=("auc", "bal_acc", "f1", "ece")):
    """One row per country, with 95% bootstrap intervals for the key metrics."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in countries:
        m = (pts["country"] == c).to_numpy()
        y, q = pts["y"].to_numpy()[m], p[m]
        r = dict(country=c, income=pts.loc[m, "income"].iloc[0],
                 continent=pts.loc[m, "continent"].iloc[0], **point_metrics(y, q))
        boots = {k: [] for k in keys}
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                bm = point_metrics(y[i], q[i])
                for k in keys:
                    boots[k].append(bm[k])
        for k in keys:
            r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile(boots[k], [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows)


def group_summary(ct, metric="auc", by="income", B=2000, seed=0):
    """Mean of a per-country metric within each group, with a bootstrap over
    countries (so a few big countries cannot dominate), and the worst country."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, d in ct.groupby(by):
        v = d[metric].dropna().to_numpy()
        bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
        worst = d.loc[d[metric].idxmin()]
        rows.append({by: g, "countries": len(v), "mean": v.mean(),
                     "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                     "worst": worst[metric], "worst_country": worst["country"]})
    out = pd.DataFrame(rows)
    if by == "income":
        out[by] = pd.Categorical(out[by], INCOME_LEVELS, ordered=True)
        out = out.sort_values(by)
    return out


def gap_ci(ct, metric="auc", hi="High", lo="Low", B=2000, seed=0):
    """High-income minus low-income mean, with a bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    a = ct.loc[ct.income == hi, metric].dropna().to_numpy()
    b = ct.loc[ct.income == lo, metric].dropna().to_numpy()
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return a.mean() - b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
