# Geographic transfer gaps in satellite cropland classification: a country-held-out benchmark

*Ashish Kesari · technical note accompanying FairEO-Bench*

## Abstract

We investigate whether crop/non-crop classifiers generalize to countries
excluded from classifier training, using NASA Harvest's CropHarvest benchmark
(87,464 labeled points across 165 countries). Six classifier configurations
are compared under country-held-out evaluation. For eligible countries,
reported average AUC differs by approximately 0.078–0.099 between low- and
high-income groups, depending on the model. A Geo-Wiki-only check retains a
difference but cannot rule out annotation noise or sampling composition.
Predicted crop prevalence differs from observed prevalence **on sampled
labeled points**; these are not national cropland-area estimates.

The income-group difference is observational, not causal. A subsequent
31-country sensitivity analysis finds an unadjusted high-minus-low AUC gap
of 0.0796 (95% country-bootstrap interval 0.0089–0.1477), compared with
0.0506 (−0.0982–0.1547) after a linear adjustment for sampled crop
prevalence. A leave-one-country-out trust-score evaluation reports
Spearman ρ = 0.72, but independent geographic validation and an audit
of upstream information reuse are still needed.

## 1. Why held-out countries

A national agency that adopts a pretrained model usually has few or no labels
of its own. The relevant question is how the model behaves on a country it
never saw. Here, random splits yield mean AUC 0.832, compared with
0.777 for country-held-out evaluation (gradient-boosted trees).

## 2. Data and protocol

- **Labels:** CropHarvest v0.7, 27 source campaigns, 2016–2021, binary crop
  label. Each point has 12 monthly steps × 18 bands.
- **Countries and income:** point-in-polygon with Natural Earth borders;
  World Bank income group as recorded in Natural Earth.
- **Splits:** 10 folds grouped by country, so each country's points are
  predicted by a model trained without that country; contrast: 10 stratified
  random folds.
- **Reporting:** 66 countries with ≥100 points and ≥20 of each class.
  Per-country AUC, balanced accuracy, recall, specificity, F1, expected
  calibration error and sampled-point crop-prevalence error. Group means use a bootstrap over
  countries (2,000 resamples); per-country intervals use a bootstrap over points.
- **Models:** LightGBM and random forest on 302 hand-made features (all monthly
  values plus annual statistics); logistic and LightGBM probes on frozen Presto
  embeddings (128-d, with and without location input); the same on frozen
  Prithvi-EO-2.0-tiny embeddings (pixel replicated to a 16×16 patch, 4 seasonal
  frames; 384-d).

## 3. Results

### 3.1 A consistent gap

See the table in the README. The gap is 0.078–0.099 AUC across models. Presto
with trees is the best model overall (0.782) and still has a gap of 0.085.
Prithvi in pixel mode is the weakest (0.696), as expected for an image model
used without spatial context.

### 3.2 Income-group AUC under different split strategies

| | Low-income AUC | High-income AUC |
|---|---|---|
| Country seen in training | 0.823 | 0.862 |
| Country held out | 0.739 | 0.819 |
| AUC difference (seen minus held out) | 0.084 | 0.043 |

(Gradient-boosted trees; Presto + trees: 0.073 and 0.035.)

### 3.3 Geo-Wiki-only labeling sensitivity check

Low-income countries draw a median 30% of their labels from Geo-Wiki,
against 92% for high-income countries. Restricting scoring to Geo-Wiki points
(56 countries) yields gaps of 0.159 (0.107–0.211, trees) and
0.171 (0.117–0.231, Presto). Using one label source does not rule out
label noise, geographic sampling differences, or other confounders.

### 3.4 Classification behavior on sampled labeled points

With the default 0.5 threshold, unseen low-income countries get recall 0.56
and specificity well above it, while high-income countries get recall 0.91.
The predicted-positive fraction minus actual crop prevalence among labeled
points is −6.6 percentage points for low-income and +9.8 for high-income
countries. These are not national area estimates without appropriate
representative sampling or spatial weighting.

### 3.5 A deployed map

ESA WorldCover (class 40 = cropland) read at each point agrees with
CropHarvest's crop label at 42–46% recall in every income group. On Geo-Wiki
labels alone, its balanced accuracy is 0.072 (0.019–0.121) lower in low-income
countries.

### 3.6 Exploratory associations with AUC

OLS on per-country AUC with robust errors (n = 66): crop share (+0.046 per SD,
p = 0.003) and log distance to the nearest labelled point in another country
(−0.045 per SD, p = 0.021) are significant; the low-income indicator falls from
−0.055 (p = 0.08) alone to −0.037 (p = 0.39) with controls. R² = 0.37.
These associations cannot identify causality. A separate 31-country
prevalence-adjusted estimate is 0.0506 (95% CI −0.0982 to 0.1547).

### 3.7 Trust score

Features: mean |p − 0.5| on the country's own (unlabelled) points, the spread of
predictions, and the mean measured AUC of the k = 5 nearest countries in
standardised Presto-embedding centroid space. Linear model, validated
leave-one-country-out: ρ = 0.72, MAE 0.061 (held-out country-mean
baseline: approximately 0.086). Of the 10 worst countries, 3 are among
the 10 flagged (trees; 4 for Presto). This internal benchmark evaluation
is not independent deployment validation.

### 3.8 Fixes

- Country-balanced sample weights (power 0.5): balanced accuracy 0.673 → 0.690
  overall and 0.659 → 0.668 in low-income countries; AUC unchanged.
- Platt recalibration on the 8 most similar countries (feature or geographic
  neighbours): no reliable gain.
- Saerens EM prior-shift: harmful in this benchmark (absolute sampled-point
  prevalence error 9.4 → 25.4 points). Poor calibration under
  geographic shift is one possible explanation.
- 20 local labels for recalibration: absolute sampled-point prevalence
  error 9.4 → 7.6 points and low-income sampled-prevalence difference
  −6.6 → −3.6 points.

### 3.9 Label budget

Retraining with 25/100/400 local labels raises a held-out country's AUC by
0.010/0.022/0.045 on average (low-income at 400: +0.056; high-income: +0.023).
Allocating 10,000 new labels: status quo 0.772 mean AUC, uniform 0.776,
novelty-guided 0.771, oracle 0.781 (from 0.752; Presto linear probe, 30k
training subsample). Even allocation beats the status quo, the simple
label-free guide does not beat even allocation, and the oracle gap is the
target for better targeting rules under the simulation's learning-curve
interpolation assumptions, not a prospective field trial.

## 4. Limitations

Point labels and mixed years; potentially noisy crop labels on small
fields; dated income classes; only 66 eligible countries; Prithvi tested
without real spatial patches; and WorldCover class definitions different
from CropHarvest's. Sampled-point prevalence cannot be promoted to
national area estimates. The trust score needs nested leakage checks
and a genuinely independent evaluation cohort. Country-held-out
classifier training alone does not ensure foundation-model pretraining
independence.

## 5. Next steps

1. Add field size (smallholder share) as an explanatory variable.
2. Run Prithvi on HLS image chips around each point, so it is used as designed.
3. Fine-tune Presto with country-balanced and group-robust objectives.
4. Turn the benchmark into an open leaderboard ranked by worst-country
   performance rather than the average.
