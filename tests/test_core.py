"""Fast checks that need no downloaded data:  python tests/test_core.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from faireo.splits import country_folds, random_folds
from faireo.metrics import point_metrics, ece, gap_ci
from faireo.fixes import fit_platt, apply_platt, em_prior_shift, balanced_weights


def fake(n=3000, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"country": r.choice(list("ABCDEFGHIJKL"), n), "y": r.integers(0, 2, n),
                         "income": "Low"})


def test_country_folds_hold_out_whole_countries():
    pts = fake()
    for tr, te in country_folds(pts, k=5):
        assert not set(pts.country.iloc[tr]) & set(pts.country.iloc[te])
    covered = np.concatenate([te for _, te in country_folds(pts, k=5)])
    assert sorted(covered) == list(range(len(pts)))


def test_random_folds_cover_everything_once():
    pts = fake()
    covered = np.concatenate([te for _, te in random_folds(pts, k=5)])
    assert sorted(covered) == list(range(len(pts)))


def test_metrics_perfect_and_chance():
    y = np.array([0, 0, 1, 1]); m = point_metrics(y, np.array([.1, .2, .8, .9]))
    assert m["auc"] == 1 and m["bal_acc"] == 1 and m["area_err_pp"] == 0
    assert abs(ece(np.array([0, 1] * 50), np.full(100, .5))) < 1e-9


def test_platt_recovers_calibration():
    r = np.random.default_rng(1); z = r.normal(0, 2, 20000); p_true = 1 / (1 + np.exp(-z))
    y = (r.random(20000) < p_true).astype(int)
    p_bad = 1 / (1 + np.exp(-(0.5 * z + 1)))          # miscalibrated
    q = apply_platt(p_bad, fit_platt(p_bad, y))
    assert ece(y, q) < ece(y, p_bad) and ece(y, q) < 0.02


def test_em_moves_prior_toward_truth():
    r = np.random.default_rng(2); y = (r.random(5000) < 0.2).astype(int)
    p = np.clip(0.5 * y + 0.25 + r.normal(0, .1, 5000), .01, .99)   # trained at prior 0.5
    _, pi = em_prior_shift(p, 0.5)
    assert abs(pi - 0.2) < abs(0.5 - 0.2)


def test_balanced_weights_equalise_countries():
    pts = pd.DataFrame({"country": ["A"] * 900 + ["B"] * 100, "y": [0, 1] * 500})
    w = balanced_weights(pts, np.arange(1000), power=1.0)
    a, b = w[:900].sum(), w[900:].sum()
    assert abs(a / b - 1) < 0.05


def test_gap_ci_sign():
    ct = pd.DataFrame({"income": ["High"] * 10 + ["Low"] * 10, "auc": [.9] * 10 + [.7] * 10})
    g, lo, hi = gap_ci(ct, "auc")
    assert abs(g - .2) < 1e-9 and lo <= g <= hi


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok  ", k); n += 1
    print(f"{n} passed")
