"""Ways to narrow the gap on countries the model has never seen.

All of these keep the test country's labels untouched unless the method name
says 'local' (few-shot), in which case only k labelled points are used and
the metrics are computed on the rest.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit


# ---------- training-time fix: give every country an equal voice ----------
def balanced_weights(pts, idx, power=0.5):
    """Weight each training point by (country size)^-power, and balance classes
    within a country. power=0 is ordinary training, power=1 makes every
    country count equally."""
    sub = pts.iloc[idx]
    size = sub.groupby("country")["y"].transform("size").to_numpy()
    pos = sub.groupby("country")["y"].transform("mean").to_numpy()
    y = sub["y"].to_numpy()
    cls = np.where(y == 1, 0.5 / np.clip(pos, 0.05, 0.95), 0.5 / np.clip(1 - pos, 0.05, 0.95))
    w = size ** (-power) * cls
    return w / w.mean()


# ---------- label-free, spatially aware recalibration ----------
def fit_platt(p, y, w=None):
    """Return (a, b) so that sigmoid(a*logit(p)+b) is calibrated on (p, y)."""
    z = logit(np.clip(p, 1e-4, 1 - 1e-4))
    w = np.ones_like(z) if w is None else w

    def nll(ab):
        q = expit(ab[0] * z + ab[1])
        return -np.sum(w * (y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9))) / w.sum()
    return minimize(nll, [1.0, 0.0], method="L-BFGS-B").x


def apply_platt(p, ab):
    return expit(ab[0] * logit(np.clip(p, 1e-4, 1 - 1e-4)) + ab[1])


def neighbour_recalibrate(pts, p, target, centroids, k=8, eligible=None):
    """Recalibrate predictions for `target` country using only the K most
    similar *other* countries (by centroid distance in `centroids`, e.g. the
    mean Presto embedding or lat/lon). Their out-of-fold predictions show how
    the model behaves on unseen places that look like the target; no target
    labels are used."""
    others = [c for c in (eligible or centroids.index) if c != target]
    d = np.linalg.norm(centroids.loc[others].to_numpy() - centroids.loc[target].to_numpy(), axis=1)
    nbrs = [others[i] for i in np.argsort(d)[:k]]
    m = pts["country"].isin(nbrs).to_numpy()
    # each neighbour country counts equally
    size = pts.loc[m].groupby("country")["y"].transform("size").to_numpy()
    ab = fit_platt(p[m], pts["y"].to_numpy()[m], 1.0 / size)
    t = (pts["country"] == target).to_numpy()
    return apply_platt(p[t], ab), nbrs


# ---------- label-free prevalence (cropland share) adjustment ----------
def em_prior_shift(p, train_prior, iters=100, tol=1e-6):
    """Saerens et al. (2002): re-estimate the crop share in a new country from
    unlabelled predictions, and adjust probabilities to that share."""
    pi = train_prior
    for _ in range(iters):
        num = p * pi / train_prior
        den = num + (1 - p) * (1 - pi) / (1 - train_prior)
        q = num / den
        new = q.mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return q, pi


# ---------- few local labels ----------
def local_recalibrate(p, y, k, rng):
    """Use k labelled points from the country to recalibrate; score the rest."""
    idx = rng.permutation(len(y))
    lab, rest = idx[:k], idx[k:]
    if y[lab].min() == y[lab].max():          # all one class: shift intercept only
        ab = np.array([1.0, logit(np.clip(y[lab].mean(), 0.02, 0.98)) - logit(np.clip(p[lab].mean(), 0.02, 0.98))])
    else:
        ab = fit_platt(p[lab], y[lab])
    return apply_platt(p[rest], ab), rest
