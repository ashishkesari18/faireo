#!/usr/bin/env python3
"""Exploratory sensitivity analysis for between-country AUC gaps.

Uses previously exported per-country metrics; does not train a model and cannot
establish causality. Both the raw and prevalence-adjusted estimates describe the
sampled, eligible countries only.

Run:
  python scripts/sensitivity_gap.py --csv results/countries_lgbm_country_s0.csv \
      --output results/validation/gap_sensitivity_lgbm_country_s0.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"country", "income", "auc", "prevalence", "n"}


def prepare(df, min_n=100, min_each=20):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Required CSV columns missing: {sorted(missing)}")
    d = df.copy()
    for col in ("auc", "prevalence", "n"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[d.income.isin(["High", "Low"])].dropna(subset=["auc", "prevalence", "n"])
    d = d[(d.n >= min_n) & (d.n * d.prevalence >= min_each) &
          (d.n * (1 - d.prevalence) >= min_each)]
    d = d[(d.auc >= 0) & (d.auc <= 1) &
          (d.prevalence >= 0) & (d.prevalence <= 1)]
    if d.country.duplicated().any():
        raise ValueError("Duplicate country rows: expected one row per country")
    if d.income.nunique() != 2 or min(d.income.value_counts()) < 3:
        raise ValueError("Need at least 3 eligible countries in each of High and Low")
    return d.reset_index(drop=True)


def raw_gap(d):
    return float(d.loc[d.income == "High", "auc"].mean() -
                 d.loc[d.income == "Low", "auc"].mean())


def prevalence_adjusted_gap(d):
    """OLS high-vs-low coefficient controlling *linearly* for sample prevalence.

    This is an observational sensitivity analysis, not a treatment effect.
    The estimator can extrapolate where income groups lack prevalence overlap.
    """
    high = (d.income.to_numpy() == "High").astype(float)
    p = d.prevalence.to_numpy(dtype=float)
    y = d.auc.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(d)), high, p])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return float("nan")
    return float(np.linalg.lstsq(x, y, rcond=None)[0][1])


def stratified_bootstrap(d, n_boot=2000, seed=2026):
    """Resample countries within each income group; CIs reflect country variation."""
    if n_boot < 100:
        raise ValueError("n_boot should be >= 100")
    rng = np.random.default_rng(seed)
    groups = [d[d.income == g] for g in ("High", "Low")]
    raws, adjs = [], []
    for _ in range(n_boot):
        b = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups],
                      ignore_index=True)
        raws.append(raw_gap(b))
        adjs.append(prevalence_adjusted_gap(b))
    def interval(x):
        v = np.asarray(x, dtype=float)
        v = v[np.isfinite(v)]
        return [float(z) for z in np.percentile(v, [2.5, 97.5])] if len(v) else None
    return interval(raws), interval(adjs)


def influence(d):
    original = raw_gap(d)
    changes = []
    for i, r in d.iterrows():
        alt = raw_gap(d.drop(index=i))
        changes.append({"country": str(r.country), "income": str(r.income),
                        "gap_without": float(alt),
                        "absolute_change": float(abs(original - alt))})
    return sorted(changes, key=lambda r: r["absolute_change"], reverse=True)[:5]


def analyze(df, n_boot=2000, seed=2026, min_n=100, min_each=20):
    d = prepare(df, min_n=min_n, min_each=min_each)
    raw_ci, adj_ci = stratified_bootstrap(d, n_boot=n_boot, seed=seed)
    high = d[d.income == "High"].prevalence
    low = d[d.income == "Low"].prevalence
    overlap = max(0., min(high.max(), low.max()) - max(high.min(), low.min()))
    return {
        "study": "Exploratory country-level sensitivity analysis; not a causal estimate",
        "filters": {"min_country_n": min_n, "min_positive_and_negative": min_each},
        "countries": int(len(d)),
        "high_countries": int(len(high)), "low_countries": int(len(low)),
        "raw_gap_high_minus_low_auc": raw_gap(d),
        "raw_country_bootstrap_95pct_ci": raw_ci,
        "prevalence_adjusted_linear_gap": prevalence_adjusted_gap(d),
        "adjusted_country_bootstrap_95pct_ci": adj_ci,
        "mean_sample_prevalence_high": float(high.mean()),
        "mean_sample_prevalence_low": float(low.mean()),
        "prevalence_range_intersection_width": float(overlap),
        "top_leave_one_country_out_influences": influence(d),
        "limitations": [
            "Country AUC and sample prevalence come from the same observational benchmark.",
            "Linear adjustment does not control all confounders or identify causal effects.",
            "The model may extrapolate if groups have little prevalence overlap.",
            "Bootstrap assumes sampled countries are exchangeable within income strata.",
            "Country counts and filters should be compared with the original publication protocol."
        ],
        "bootstrap_seed": seed, "bootstrap_replicates": n_boot
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/countries_lgbm_country_s0.csv")
    parser.add_argument("--output", default="results/validation/gap_sensitivity_lgbm_country_s0.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        parser.error(f"Missing input {path}. Run the original pipeline first.")
    result = analyze(pd.read_csv(path), n_boot=args.bootstrap, seed=args.seed)
    path_out = Path(args.output)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {path_out} (generated from real CSV; inspect before publishing)")


if __name__ == "__main__":
    main()
