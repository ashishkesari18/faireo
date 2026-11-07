# Experiment protocol for review

**Prepared:** 2026-11-07. **Status:** pre-execution plan; no claim that these experiments have been run on FairEO data.

## E0 — Basic structural consistency checks

**Question:** Are the exported country metrics arithmetically consistent?

```bash
python scripts/verify_country_results.py --input results/countries_lgbm_country_s0.csv
```

**Pass criteria:** Each country is unique; `n` positive; crop and predicted crop prevalence within [0,1]; AUC within [0,1] where defined; and `area_err_pp` matches `100*(pred_prevalence-prevalence)` up to numerical tolerance. This only proves internal consistency, not scientific correctness.

## E1 — Country gap sensitivity to sampled crop prevalence

**Question:** Does the observed high-minus-low macro-country AUC gap remain directionally similar after a simple linear adjustment for sample crop prevalence? How influential is any one country?

```bash
python scripts/sensitivity_gap.py \
  --csv results/countries_lgbm_country_s0.csv \
  --output results/validation/gap_sensitivity_lgbm_country_s0.json \
  --bootstrap 2000 --seed 2026
```

**Inputs:** one CSV of per-country observations, not individual points. Preserve minimum n>=100 and at least 20 cases in each class. Compare across only the `Low` and `High` income groups (do not silently collapse intermediate groups).

**Outputs to inspect:** eligible-country counts, raw and linear adjusted gap, both 95% country-bootstrap intervals, overlap of prevalence ranges, top five leave-one-out influences.

**Interpretation rule:** If adjusted and raw gaps differ materially, investigate prevalence confounding. If prevalence overlap is weak, do not present the adjusted coefficient as robust. In any outcome, do not make causal claims. Register further sensitivity checks before trying them.

## E2 — Rerun predictions with a different split seed (owner-run)

**Question:** Are conclusions dominated by one country-fold assignment?

```bash
python scripts/audit.py --model lgbm --split country --seed 1
python scripts/report.py
```

Record the *new* CSV or prediction file generated, relevant dependency versions, and changed country-gap statistics. Verify that `scripts/report.py` processes the new file as expected, before comparing. Do not claim this experiment is complete until the actual output is inspected.

## E3 — Model and trust-score leakage review (manual + code)

Trace every data transformation from original observations through features, split generation, model fitting, tuning, similarity scoring, and calibration. For each fitted transformation write: (a) what it learned, (b) which samples it saw, (c) what is available at inference, and (d) whether held-out countries or labels affected it. If any preprocessing depends on all countries, distinguish valid unsupervised transductive access from an intended inductive setting and document it explicitly.

## E4 — External model evaluation (future)

Collect true georeferenced image chips for Prithvi rather than pixel replication. Pre-register new-country, out-of-year, sensor, and labeling-protocol strata. Run an independent evaluation that cannot be tuned from the current benchmark labels.

## Reproducibility checklist

- [ ] Save `git rev-parse HEAD` and branch
- [ ] Save Python version and `pip freeze`
- [ ] Save SHA256 digests for input tables and downloaded weights
- [ ] Run original core tests and reviewer-added synthetic tests
- [ ] Run E0 and E1 on actual repository CSVs
- [ ] Verify baseline runs from the actual data, not only saved country summaries
- [ ] Add the generated outputs and observed findings to the dated research log
- [ ] Review wording and figures before publicizing new claims
