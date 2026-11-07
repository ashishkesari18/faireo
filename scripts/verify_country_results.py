#!/usr/bin/env python3
"""Check internal consistency of an exported per-country metrics table.

This is not a substitute for rerunning models or validating labels. We check
quantities that must agree mathematically within the *sampled labeled points*.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"country", "n", "prevalence", "pred_prevalence", "area_err_pp", "auc"}


def verify(df, tolerance=0.0001):
    problems = []
    missing = REQUIRED - set(df.columns)
    if missing:
        return [f"Missing columns: {sorted(missing)}"]
    if df.country.isna().any() or df.country.duplicated().any():
        problems.append("Missing or duplicated country identifier")
    for col in ("n", "prevalence", "pred_prevalence", "area_err_pp", "auc"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Non-numeric column {col}")
    if problems:
        return problems
    if (df.n <= 0).any() or df.n.isna().any():
        problems.append("Non-positive or missing sample count")
    for col in ("prevalence", "pred_prevalence"):
        if (~df[col].between(0, 1)).any() or df[col].isna().any():
            problems.append(f"Invalid {col} outside [0, 1]")
    if (~df.auc.dropna().between(0, 1)).any():
        problems.append("AUC outside [0, 1]")
    expected = 100 * (df.pred_prevalence - df.prevalence)
    if not np.allclose(expected, df.area_err_pp, atol=tolerance, rtol=0, equal_nan=False):
        problems.append("area_err_pp != 100 * (pred_prevalence - prevalence)")
    if df.area_err_pp.isna().any():
        problems.append("Missing area_err_pp")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/countries_lgbm_country_s0.csv")
    args = p.parse_args()
    source = Path(args.input)
    if not source.is_file():
        p.error(f"No file {source}")
    df = pd.read_csv(source)
    issues = verify(df)
    if issues:
        print(f"FAIL: {source}")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"PASS: {len(df)} country rows in {source} satisfy basic metric invariants")
    print("NOTE: This does not validate cropland area, data quality, or scientific findings.")


if __name__ == "__main__":
    main()
