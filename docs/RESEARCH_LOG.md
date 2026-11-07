# FairEO-Bench research log

> **Status: ongoing research record (updated 2026-11-07).** Entries distinguish desk-based reviews, locally executed data checks, and experiments that remain planned. Model-training reproduction is still pending.

## Research question

When a cropland classification model is evaluated on countries whose samples were excluded from training, how much does its performance vary across countries and World Bank income groups? Which observable covariates may account for that variation? Can the model's reliability be forecast *without target-country labels*?

## Research claims to investigate, not assume

1. Country-held-out evaluation exposes geographic generalization failures that random, point-level splits may conceal.
2. The high-income minus low-income macro-average AUC difference remains after country-level controls for sampled crop prevalence and label protocol. **Needs independent replication and sensitivity analysis.**
3. An unlabeled trust score can predict AUC in new countries. **Needs genuinely independent validation and evaluation of nested training / feature construction.**
4. Adding local labels improves performance in difficult countries. **Needs confirmation that test points are excluded from local training and hyperparameter selection.**

## 2026-11-07 — Desk review; no new model run

**What was inspected:** public `README.md`, `faireo/metrics.py`, `scripts/audit.py`, `tests/test_core.py`, and `.github/workflows/tests.yml`.

**Methodological concern:** `point_metrics` calculates `area_err_pp = 100 * (pred.mean() - y.mean())` on collected labeled points. Without an area-representative sample or appropriate inclusion/design weights, this is sampled-point prevalence error, **not** national cropland-area error. Keep the column name for backwards compatibility for now; correct the interpretation.

**Evaluation concern:** country held-out folds are a useful design, but correlations and bootstrap intervals do not identify causal mechanisms. Need to inspect how feature preparation, tuning, trust-score training, and data re-use are separated from held-out targets.

**Implementation concern:** CI currently runs basic tests but does not reconstruct model predictions or compare exported metrics with the underlying labeled observations.

**Change proposed:** add a standalone country-level sensitivity analysis that reuses saved metrics and explicitly reports both raw and linear prevalence-adjusted AUC gaps. This is a diagnostic only, not a causal estimate.

**Execution status at this desk-review stage:** four synthetic review-pack
unit tests had passed. The complete model-training pipeline was not
reproduced. Later saved-CSV checks are recorded below.

## Template for future experiment entries

- Date and operator:
- Git commit SHA and branch:
- Environment and `python --version`:
- Data version / CSV checksum:
- Research hypothesis **written before seeing result**:
- Exact commands:
- New artifacts generated:
- Difference from existing reported results:
- Interpretation / alternative explanations:
- Unexpected failures or negative results:
- What was verified by the researcher:
- Next test:

## Decisions the owner must make

- What is the primary claim to prioritize: geographic evaluation, failure forecasting, or annotation allocation?
- What is the intended deployment unit: a pixel, field, district, or country?
- What criteria would lead to rejecting the claimed income gap or the trust score?
- Which results can be reproduced from raw CropHarvest observations, not just previously saved summary CSVs?

## 2026-11-07 — Country-level sensitivity analysis

Research question:
Does the observed income-group AUC gap remain
after adjusting for sampled crop prevalence?

Data:
results/countries_lgbm_country_s0.csv

Validation:
66 country rows passed basic metric-invariant checks.

Analysis:
31 countries met the sensitivity-analysis filters:
10 high-income and 21 low-income.

Results:
- Raw AUC gap: 0.07964
- Raw bootstrap 95% CI: [0.00893, 0.14771]
- Prevalence-adjusted gap: 0.05063
- Adjusted bootstrap 95% CI: [-0.09823, 0.15473]
- Bootstrap replicates: 2,000

Interpretation:
The unadjusted gap is positive, but the
prevalence-adjusted interval includes zero.
These observational results do not establish
a causal income-group effect.

Limitations:
Only 31 countries met the analysis filters.
Prevalence adjustment is linear and incomplete.
Country-level bootstrap assumptions need
further examination.

Next experiment:
Evaluate robustness across additional seeds,
country eligibility thresholds, and other
sampling-composition variables.

Documentation note:
These results were computed from stored per-country
CSV outputs rather than independent model retraining.
Scientific interpretation and further verification remain open.

## 2026-11-07 — Public claim and map-display audit

- Revised unsupported national-area and causal interpretations in public text.
- Identified and corrected use of the evaluated country's observed AUC
  in the original constant-mean trust-score baseline.
- Updated public trust-map generation to display the saved leave-one-country-out
  estimate for measured countries, instead of predictions from the
  trust model fitted on all measured-country outcomes.
- External geographic validation and end-to-end model reproduction remain open.
