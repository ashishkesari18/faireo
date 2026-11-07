"""Model zoo. Each entry: (feature builder, classifier factory)."""
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DATA, tabular


def _presto(key, file="presto_emb.npz"):
    def f(X, pts):
        return np.load(DATA / file)[key]
    return f


def _tab(X, pts):
    return tabular(X)


MODELS = {
    "rf": (_tab, lambda: RandomForestClassifier(200, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=0)),
    "lgbm": (_tab, lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                             subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                             verbose=-1, random_state=0)),
    "presto_lr": (_presto("emb"), lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
    "presto_lgbm": (_presto("emb"), lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                              num_leaves=63, verbose=-1, random_state=0)),
    "presto_noloc_lr": (_presto("emb_noloc"), lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
    "prithvi_lr": (_presto("emb", "prithvi_emb.npz"), lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))),
    "prithvi_lgbm": (_presto("emb", "prithvi_emb.npz"), lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                                                  num_leaves=63, verbose=-1, random_state=0)),
}
