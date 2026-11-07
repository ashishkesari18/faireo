"""Synthetic tests for the reviewer-added country-level analysis (no data download)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.sensitivity_gap import prepare, raw_gap, prevalence_adjusted_gap, analyze
from scripts.verify_country_results import verify


def fake_countries():
    p = np.linspace(0.2, 0.8, 12)
    high_p = np.clip(p + .05, 0, 1)
    low_p = p
    return pd.DataFrame({
        "country": [f"H{i}" for i in range(12)] + [f"L{i}" for i in range(12)],
        "income": ["High"] * 12 + ["Low"] * 12,
        "n": [200] * 24,
        "prevalence": list(high_p) + list(low_p),
        "auc": list(.60 + .05 * high_p + .04) + list(.60 + .05 * low_p)
    })


def test_adjusted_gap_recovers_known_coefficient():
    d = prepare(fake_countries())
    assert abs(prevalence_adjusted_gap(d) - .04) < 1e-10
    assert abs(raw_gap(d) - .0425) < 1e-10


def test_bootstrap_and_influence():
    out = analyze(fake_countries(), n_boot=120, seed=11)
    assert out["countries"] == 24
    assert np.isfinite(out["prevalence_adjusted_linear_gap"])
    assert out["raw_country_bootstrap_95pct_ci"][0] <= out["raw_country_bootstrap_95pct_ci"][1]
    assert len(out["top_leave_one_country_out_influences"]) == 5


def test_country_table_invariants():
    d = pd.DataFrame({"country": ["A", "B"], "n": [10, 20],
                      "prevalence": [.4, .5], "pred_prevalence": [.5, .35],
                      "area_err_pp": [10., -15.], "auc": [.7, .8]})
    assert verify(d) == []
    d.loc[0, "area_err_pp"] = 11
    assert verify(d)


def test_does_not_infer_unsupported_population_area():
    # A balanced case/control sample can have 50% sampled prevalence regardless
    # of true land-area cropland proportion. Area cannot be inferred from these rows.
    d = pd.DataFrame({"country": ["A"], "n": [100], "prevalence": [.5],
                      "pred_prevalence": [.6], "area_err_pp": [10.], "auc": [.8]})
    assert verify(d) == []


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
            passed += 1
    print(f"{passed} review-pack tests passed")
