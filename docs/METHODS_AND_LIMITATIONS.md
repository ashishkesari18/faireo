# Methods, validity boundaries, and intended interpretation

> Methods note (updated 2026-11-07). Arithmetic checks of 66 saved country
> rows and an exploratory 31-country sensitivity analysis were executed.
> Neither is an independent reproduction of model training.

## Unit of observation versus unit of inference

The benchmark's labels are observations at geographic points. Its country-level AUCs average over those labeled points. Cross-country gap estimates then average eligible countries, giving each country equal macro-level weight. These choices answer useful questions about the **benchmark sample**, but not necessarily about every pixel, farm, or hectare in each country.

AUC measures ranking on the evaluation sample. It is not an error measure for national cultivated area. In the existing metrics code, the legacy `area_err_pp` field is `100 * (predicted-positive fraction - actual-positive fraction)` across sampled points. It is a **percentage-point sampled-prevalence difference**. National cropland area would additionally require spatially representative sampling or defensible design-based weights, an area denominator, harmonized labels, and relevant error propagation.

## Generalization

`country` splits exclude all labeled target-country observations from training within each fold. They test out-of-country generalization, unlike `random` splits that can contain observations from the same country on both sides. Cross-country generalization is still not equivalent to out-of-time or out-of-sensor generalization. Spatial autocorrelation and campaign clustering can persist among countries.

Avoid concluding that a measured high/low-income difference *comes from* economic status or a particular causal mechanism. Income is a proxy entangled with land cover, field size, sensor quality, geography, campaign protocols, and sample prevalence. Country-level regression coefficients are associations.

## Paired country-level sensitivity analysis

`scripts/sensitivity_gap.py` takes the saved per-country AUC table and:

- applies the stated minimum-country sample and class-count eligibility thresholds;
- reports the unadjusted high-minus-low macro-country AUC gap with country-stratified bootstrap intervals;
- reports a **linear prevalence-adjusted** coefficient with bootstrap intervals;
- reports the common range width for *sampled* crop prevalence, to flag extrapolation risk;
- reports the five most influential countries for the unadjusted gap under leave-one-country-out removal.

Interpret the adjusted value as a **sensitivity check**, not a bias-corrected causal effect. The adjustment is linear, may extrapolate when groups have little covariate overlap, and does not adjust other confounders. Bootstrapping countries assumes within-group exchangeability and may understate uncertainty from correlated sampling campaigns or geographic clusters. Pre-specify comparisons before choosing models or filtering outcomes.

This analysis reuses stored per-country outputs. It **does not independently reproduce the trained models**. Validate generated metrics against raw out-of-fold predictions, dataset checksums, and model/checkpoint versions before publication.

## Model-specific limitations

**Presto:** Verify the exact weight release, feature preprocessing, temporal window, and whether embedding models had overlapping training data with the evaluation dataset. A country holdout during classifier training alone does not guarantee pretraining independence.

**Prithvi:** The README indicates that each pixel is duplicated over a 16×16 patch for model input. Because an image model expects spatial context, that adaptation is not a fair comparison with actual image patches. Re-run with real, georeferenced, multiband HLS image chips before comparing model quality.

**Geo-Wiki and WorldCover:** Label definitions, years, resolution, and sampling protocols differ. A subset with a common labeling protocol is an informative sensitivity analysis but cannot rule out labeling noise, spatial bias, or domain mismatch. WorldCover is not independent ground truth for all crop definitions.

**Trust score:** Distinguish predictor information available at deployment (unlabeled features) from information learned from the held-out country or from its labels. Audit nested cross-validation of score fitting, feature selection, tuning, and neighborhood selection. A leave-one-country-out score fit is not sufficient if the upstream model, hyperparameters, or similarity descriptors have seen the evaluation outcomes. The map must show `expected_auc_loo` for measured countries; `expected_auc` is fitted with all measured-country outcomes and is only appropriate for sparse countries excluded from that fit. The constant-mean comparison baseline must likewise exclude the evaluated country.

## Language to use until validation is stronger

| Avoid | More supportable |
|---|---|
| "Cropland is under-counted nationally" | "Positive-class prevalence is underestimated on the sampled labeled points" |
| "Poverty causes poorer satellite AI" | "In this benchmark, model performance is associated with income-group classification" |
| "Label bias is ruled out" | "The gap persists under a Geo-Wiki-only subset, subject to remaining confounders" |
| "Foundation model A is better than B" | "Under the stated feature and pixel-adaptation protocols, classifier A has higher sample AUC" |
| "Trust score predicts real-world reliability" | "Trust score is associated with held-out benchmark-country AUC; external validation is pending" |

## Minimum conditions for a paper submission

1. Dataset provenance and eligibility counts by group, campaign, and country are reproducible.
2. Country-held-out folds, models, and outputs can be regenerated from scripts with pinned dependency versions and archived source checksums.
3. Country bootstrap and alternative geographic-cluster resampling yield reasonably compatible uncertainty assessments.
4. The trust score has a nested, label-free target-country evaluation and relevant simple baselines.
5. The sampled-versus-area distinction is resolved in every figure, table, and interactive-map layer.
6. Known limitations and null experiments are presented without causal overclaiming.
