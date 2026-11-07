# Research experiment: continent-held-out reliability stress test

## Research question

Can country-level classifier performance be estimated from **unlabeled
prediction statistics** when an entire continent has been excluded from
fitting the reliability model?

This is a more geographically stringent **calibrator** test than
leave-one-country-out. It is **not** a continent-held-out retraining of
the underlying crop classifier or an independent external-data validation.

## Data, provenance, and constraints

Use the 66 eligible measured countries in each of:

- results/countries_lgbm_country_s0.csv and results/trust_lgbm_country_s0.csv
- results/countries_presto_lgbm_country_s0.csv and results/trust_presto_lgbm_country_s0.csv

The stored classifier out-of-fold predictions were generated using
country-held-out folds. Consequently, upstream training may have included
labels from other countries in the test continent. The saved confidence
statistics inherit this limitation. No claim is made that the
*entire model pipeline* is independent of the held-out continent.

All numerical findings are exploratory: the country dataset and trust-score
research already existed before this stress test was designed.

## Fixed evaluation protocol

1. Group countries by the recorded continent. Six continents remain:
   Africa, Asia, Europe, North America, Oceania, South America.
2. For each fold, fit a reliability model using AUC labels from countries
   **outside** the test continent; predict all eligible countries in the
   held-out continent. Every country is tested exactly once.
3. Evaluate a training-continent mean AUC baseline.
4. Evaluate a **confidence-only** ridge regression.
5. Evaluate a fixed-alpha **three-feature** ridge regression:
   mean absolute prediction confidence, prediction spread, and
   predicted-positive crop fraction. All features can be computed
   without observed target-country crop labels.
6. Standardize features on the training continents only. Use ridge
   alpha = 10.0; no fitting, model selection, or tuning on test-continent AUC.
7. Evaluate country-macro MAE (primary), RMSE, Spearman rank correlation,
   detection overlap in the lowest 10% AUC countries, and per-continent and
   per-income MAE.
8. Report a paired MAE improvement (baseline minus model) and an
   exploratory 95% percentile interval from 4,000 resamples of the **six
   continents** (seed 2026). This bootstrap is fragile with six clusters,
   particularly when group sizes are imbalanced.

**Interpretation rule:** A lower held-out continent MAE provides evidence
that these unlabeled *country summary features* contain predictive signal
beyond the training-continent mean baseline. It does not establish external
validity, fair deployment behavior, causal geographic mechanisms, or
that the original trust score transfers to newly sampled continents.

The original published trust-score leave-one-country-out MAE is shown
separately for historical context only. It is **not** directly comparable
to continent-held-out estimates, because its validation folds are different
and it uses embedding-space neighbor AUC information unavailable for
a continent-blocked re-fit from these saved tables.

## Reproduce locally

Install the existing project requirements, then run:

    python scripts/geographic_stress_test.py
    python tests/test_geographic_stress.py

This writes:

- results/validation/continent_holdout_stress.json
- results/validation/continent_holdout_predictions.csv

The script records SHA256 digests for each input CSV and outputs held-out
predictions and grouped diagnostics. GitHub Actions runs these checks and
uploads both generated files.

## Conditions for the next, truly independent experiment

1. Reconstruct the entire upstream feature and prediction pipeline with
   the target **continent** excluded from classifier training, tuning,
   normalization, supervised pretraining adaptations, and neighborhood
   selection (as appropriate to each model).
2. Use independent georeferenced image patches for image-native models.
3. Freeze the trust-score design before seeing held-out outcomes.
4. Test on new label sources, years, or geographic regions not used in
   any model or analysis choices.
5. Predefine a failure criterion; for example, no positive MAE improvement
   over a properly blocked baseline, or deterioration in priority regions.

Only then evaluate whether the method is ready to support deployment
decisions.
