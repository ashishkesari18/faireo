"""Train/test splits. 'country' holds out whole countries; 'random' does not."""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def country_folds(pts, k=10, seed=0):
    """Every country lands in exactly one test fold, so each point gets a
    prediction from a model that never saw any point from its country."""
    rng = np.random.default_rng(seed)
    countries = pts["country"].unique()
    # spread big countries across folds so folds are similar in size
    sizes = pts["country"].value_counts()
    order = sorted(countries, key=lambda c: (-sizes[c], rng.random()))
    load = np.zeros(k)
    assign = {}
    for c in order:
        f = int(np.argmin(load + rng.random(k) * 1e-6))
        assign[c] = f
        load[f] += sizes[c]
    fold = pts["country"].map(assign).to_numpy()
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def random_folds(pts, k=10, seed=0):
    """Ordinary random split: the model has seen other points from the same country."""
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    strata = pts["country"].astype(str) + "_" + pts["y"].astype(str)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= k, "rare")
    return list(skf.split(np.zeros(len(pts)), strata))


SPLITS = {"country": country_folds, "random": random_folds}
