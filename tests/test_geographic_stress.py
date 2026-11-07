"""Synthetic regression tests for continent-blocked country-level validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.geographic_stress_test import (
    FEATURE_SETS, load_countries, predict_heldout_continents,
    cluster_paired_ci, summarize
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def toy():
    rows = []
    for continent, delta in [("Africa", -0.04), ("Asia", 0.0),
                             ("Europe", 0.05)]:
        for i in range(8):
            conf = 0.18 + i * 0.02
            rows.append({
                "country": continent + str(i), "continent": continent,
                "income": "Low" if i % 2 else "High",
                "auc": 0.55 + conf * .45 + delta,
                "confidence": conf, "spread": .12 + .012 * i,
                "pred_crop_share": .2 + .07 * i,
                "expected_auc_loo": .70
            })
    return pd.DataFrame(rows)


def test_every_country_held_out_as_entire_continent():
    d = toy()
    p = predict_heldout_continents(d)
    assert len(p) == len(d)
    for continent in d.continent.unique():
        train_mean = d.loc[d.continent != continent, "auc"].mean()
        assert np.allclose(p.loc[p.continent == continent, "mean_baseline"], train_mean)


def test_no_target_continent_outcome_used():
    d = toy()
    a = predict_heldout_continents(d)
    d.loc[d.continent == "Africa", "auc"] += .15
    b = predict_heldout_continents(d)
    for col in ("mean_baseline", "confidence_only", "three_feature"):
        original = a.loc[a.continent == "Africa", col].to_numpy()
        edited = b.loc[b.continent == "Africa", col].to_numpy()
        assert np.allclose(original, edited), col


def test_input_features_are_not_overwritten_by_predictions():
    d = toy()
    p = predict_heldout_continents(d)
    expected = d.set_index("country")["confidence"].reindex(p.country).to_numpy()
    assert np.allclose(p.confidence, expected)
    assert "confidence_only" in p.columns
    assert "confidence" not in FEATURE_SETS


def test_three_feature_predictions_use_original_confidence():
    d = toy()
    p = predict_heldout_continents(d)
    train = d[d.continent != "Europe"]
    test = d[d.continent == "Europe"].sort_values("country")
    scaler = StandardScaler().fit(train[FEATURE_SETS["three_feature"]])
    model = Ridge(alpha=10.0).fit(
        scaler.transform(train[FEATURE_SETS["three_feature"]]),
        train.auc.to_numpy(dtype=float)
    )
    expected = model.predict(scaler.transform(test[FEATURE_SETS["three_feature"]]))
    actual = p[p.continent == "Europe"].sort_values("country").three_feature.to_numpy()
    assert np.allclose(actual, expected)


def test_cluster_bootstrap_deterministic():
    p = predict_heldout_continents(toy())
    x = cluster_paired_ci(p, "three_feature", B=200, seed=101)
    assert x == cluster_paired_ci(p, "three_feature", B=200, seed=101)
    assert len(x) == 2 and x[0] <= x[1]
    summary = summarize(p, B=200, seed=101)
    assert summary["n_countries"] == 24
    assert summary["n_continents"] == 3


def test_reject_inconsistent_original_results(tmp_path):
    d = toy()
    ct = d[["country", "continent", "income", "auc"]]
    tr = d[["country", "confidence", "spread", "pred_crop_share",
            "expected_auc_loo"]].copy()
    tr["measured_auc"] = d.auc.to_numpy() + .10
    a, b = tmp_path / "country.csv", tmp_path / "trust.csv"
    ct.to_csv(a, index=False)
    tr.to_csv(b, index=False)
    try:
        load_countries(a, b)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("An inconsistent source table should be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if name == "test_reject_inconsistent_original_results":
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("ok", name)
