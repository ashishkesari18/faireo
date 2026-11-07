#!/usr/bin/env python3
"""Exploratory leave-one-continent-out reliability stress test.

This is a country-level *calibrator* evaluation on previously saved
out-of-fold classifier predictions, not an end-to-end continent-held-out
retraining experiment. No target-continent country AUC enters the
calibrator fit or the constant baseline in a fold.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("lgbm_country_s0", "presto_lgbm_country_s0")
FEATURE_SETS = {
    "confidence": ["confidence"],
    "three_feature": ["confidence", "spread", "pred_crop_share"],
}
ALPHA = 10.0
SEED = 2026


def load_countries(country_csv, trust_csv):
    """Require a one-to-one join against stored, eligible country AUCs."""
    countries = pd.read_csv(country_csv)
    trust = pd.read_csv(trust_csv)
    required_c = {"country", "continent", "income", "auc"}
    required_t = {"country", "confidence", "spread", "pred_crop_share",
                  "measured_auc", "expected_auc_loo"}
    if not required_c.issubset(countries) or not required_t.issubset(trust):
        raise ValueError("Missing required country or trust-score columns")
    if (countries.country.isna().any() or trust.country.isna().any()
            or countries.country.duplicated().any()
            or trust.country.duplicated().any()):
        raise ValueError("Country identifiers must be nonmissing and unique")
    d = countries[["country", "continent", "income", "auc"]].merge(
        trust[["country", "confidence", "spread", "pred_crop_share",
               "measured_auc", "expected_auc_loo"]],
        on="country", how="left", validate="one_to_one", indicator=True)
    if not (d._merge == "both").all():
        raise ValueError("Missing trust observations for eligible countries")
    if d[["continent", "income"]].isna().any().any():
        raise ValueError("Missing geographic group metadata")
    numeric = ["auc", "confidence", "spread", "pred_crop_share",
               "measured_auc", "expected_auc_loo"]
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if not np.isfinite(d[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite value in eligible country results")
    if not np.allclose(d.auc, d.measured_auc, atol=1e-8, rtol=0):
        raise ValueError("Trust CSV and metric CSV disagree on actual country AUC")
    if not d.auc.between(0, 1).all():
        raise ValueError("AUC outside [0, 1]")
    if not d.pred_crop_share.between(0, 1).all():
        raise ValueError("Predicted crop share outside [0, 1]")
    if d.continent.nunique() < 3:
        raise ValueError("At least three geographic groups needed")
    return d.drop(columns="_merge").sort_values("country").reset_index(drop=True)


def predict_heldout_continents(d, alpha=ALPHA):
    """Standardize and fit entirely within the training continents."""
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    output = []
    for continent in sorted(d.continent.unique()):
        train = d[d.continent != continent]
        test = d[d.continent == continent].copy()
        if len(train) < 5:
            raise ValueError("Too few training countries in held-out split")
        test["mean_baseline"] = float(train.auc.mean())
        for name, features in FEATURE_SETS.items():
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(train[features].to_numpy(dtype=float))
            Xte = scaler.transform(test[features].to_numpy(dtype=float))
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(Xtr, train.auc.to_numpy(dtype=float))
            test[name] = model.predict(Xte)
        output.append(test)
    pred = pd.concat(output, ignore_index=True)
    if (len(pred) != len(d) or pred.country.duplicated().any()
            or pred[["mean_baseline", *FEATURE_SETS]].isna().any().any()):
        raise AssertionError("Held-out country coverage is incomplete")
    return pred.sort_values("country").reset_index(drop=True)


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    k = max(1, int(np.ceil(len(y) / 10)))
    worst = set(np.argsort(y)[:k])
    flagged = set(np.argsort(pred)[:k])
    rho = float(spearmanr(y, pred).statistic)
    return {
        "n": int(len(y)),
        "mae": round(mae(y, pred), 6),
        "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 6),
        "spearman_rho": round(rho, 6) if np.isfinite(rho) else None,
        "worst_10pct_overlap": int(len(worst & flagged)),
        "worst_10pct_k": int(k),
    }


def cluster_paired_ci(pred, candidate, B=4000, seed=SEED):
    """Percentile CI on baseline-minus-candidate MAE, resampling continents.

    With six continents and unbalanced group sizes, this is descriptive,
    not a definitive inferential test.
    """
    if B < 100:
        raise ValueError("At least 100 bootstrap draws required")
    groups = [p for _, p in pred.groupby("continent", sort=True)]
    ns = np.array([len(p) for p in groups], dtype=float)
    baseline_sums = np.array([
        np.abs(p.auc - p.mean_baseline).sum() for p in groups
    ], dtype=float)
    candidate_sums = np.array([
        np.abs(p.auc - p[candidate]).sum() for p in groups
    ], dtype=float)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(groups), size=(B, len(groups)))
    delta = (baseline_sums[selected].sum(axis=1)
             - candidate_sums[selected].sum(axis=1)) / ns[selected].sum(axis=1)
    return [round(float(v), 6) for v in np.percentile(delta, [2.5, 97.5])]


def summarize(pred, B=4000, seed=SEED):
    baseline_mae = mae(pred.auc, pred.mean_baseline)
    methods = {}
    for name in ("mean_baseline", "confidence", "three_feature"):
        result = score(pred.auc, pred[name])
        if name != "mean_baseline":
            result["mae_improvement_over_baseline"] = round(
                baseline_mae - result["mae"], 6)
            result["continent_cluster_95pct_ci_improvement"] = cluster_paired_ci(
                pred, name, B=B, seed=seed)
        methods[name] = result
    return {
        "n_countries": int(len(pred)),
        "n_continents": int(pred.continent.nunique()),
        "methods": methods,
        "published_country_loo_reference_not_comparable": score(
            pred.auc, pred.expected_auc_loo),
        "by_continent": [
            {"continent": str(continent), "n": int(len(p)),
             "mean_baseline_mae": round(mae(p.auc, p.mean_baseline), 6),
             "three_feature_mae": round(mae(p.auc, p.three_feature), 6)}
            for continent, p in pred.groupby("continent", sort=True)
        ],
        "by_income": [
            {"income": str(income), "n": int(len(p)),
             "mean_baseline_mae": round(mae(p.auc, p.mean_baseline), 6),
             "three_feature_mae": round(mae(p.auc, p.three_feature), 6)}
            for income, p in pred.groupby("income", sort=True)
        ],
    }


def run(root=ROOT, B=4000, seed=SEED):
    output = {
        "status": "exploratory_saved_results_geographic_stress_test",
        "study_date": "2026-11-07",
        "design": "Leave one entire continent out when fitting country-level AUC calibrators",
        "fixed_alpha": ALPHA,
        "features": FEATURE_SETS,
        "bootstrap": {"unit": "continent", "replicates": B, "seed": seed},
        "important_scope": [
            "The underlying classifier out-of-fold predictions were originally trained with country-held-out folds, not continent-held-out retraining.",
            "The confidence, spread, and predicted-positive share are derived from those already saved classifier predictions.",
            "This test excludes target-continent AUC labels from calibrator fitting but cannot establish independence of the upstream classifier or feature representations.",
            "The original published trust-score leave-one-country-out results are supplied for context only, not as a directly comparable continent-holdout competitor.",
            "The existing 66-country benchmark was already used during project development: treat results as exploratory, not confirmatory or external validation.",
            "Only six continents, with particularly small North America and Oceania groups; cluster bootstrap confidence intervals require caution."
        ],
        "runs": {}
    }
    frames = []
    for run_name in RUNS:
        country_csv = Path(root) / "results" / ("countries_" + run_name + ".csv")
        trust_csv = Path(root) / "results" / ("trust_" + run_name + ".csv")
        d = load_countries(country_csv, trust_csv)
        p = predict_heldout_continents(d)
        output["runs"][run_name] = {
            "country_csv_sha256": hashlib.sha256(country_csv.read_bytes()).hexdigest(),
            "trust_csv_sha256": hashlib.sha256(trust_csv.read_bytes()).hexdigest(),
            **summarize(p, B=B, seed=seed),
        }
        p["run"] = run_name
        frames.append(p[["run", "country", "continent", "income", "auc",
                         "mean_baseline", "confidence", "three_feature",
                         "expected_auc_loo"]].rename(
                             columns={"auc": "observed_auc",
                                      "expected_auc_loo": "published_loo_reference"}))
    return output, pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="results/validation/continent_holdout_stress.json")
    ap.add_argument("--predictions", default="results/validation/continent_holdout_predictions.csv")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    report, pred = run(B=args.bootstrap, seed=args.seed)
    out = ROOT / args.output
    csv = ROOT / args.predictions
    out.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pred.to_csv(csv, index=False, float_format="%.9f")
    print("BEGIN_CONTINENT_HOLDOUT_RESULT")
    print(json.dumps(report, indent=2, allow_nan=False))
    print("END_CONTINENT_HOLDOUT_RESULT")
    print("Predictions saved:", csv)


if __name__ == "__main__":
    main()
