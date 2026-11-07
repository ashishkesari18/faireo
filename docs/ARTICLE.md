# Cropland AI under-counts farms where it has never been: a country-held-out audit of satellite models

*Ashish Kesari · technical note accompanying FairEO-Bench*

## Abstract

Satellite foundation models are becoming the default starting point for
national cropland and food-security maps, but they are mostly evaluated on
random splits that mix every country into training and test. We audit six
crop/non-crop classifiers on NASA Harvest's CropHarvest dataset (87,464
labelled points, 165 countries) with every country held out of training. The
classifiers are tree ensembles on raw monthly Sentinel-1/2, ERA5 and SRTM
series, and probes on frozen Presto and Prithvi-EO-2.0 embeddings. Across all
six, per-country AUC is 0.07–0.10 lower in low-income than in high-income
countries, and every 95% interval excludes zero. The gap roughly halves when
countries are seen in training, and it grows to 0.16–0.17 when only labels
from a single worldwide protocol (Geo-Wiki) are scored, so it is not an
artefact of local labelling campaigns. Errors are one-sided: in unseen
low-income countries the models miss crop (recall 0.54–0.56 against
0.89–0.91 in high-income countries) and under-state cropland share by 6.6–8.3
percentage points. We find that crop share and distance to labelled data
explain most of the income effect, that post-hoc recalibration without local
labels does not close the gap, and that a label-free trust score predicts
per-country reliability (Spearman ρ = 0.72, leave-one-country-out).

## 1. Why held-out countries

A national agency that adopts a pretrained model usually has few or no labels
of its own. The relevant question is how the model behaves on a country it
never saw, and random splits overstate that badly: mean AUC is 0.832 with
countries seen, against 0.777 with them held out (gradient-boosted trees).

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
  calibration error and cropland-share error. Group means use a bootstrap over
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

### 3.2 Being unseen hurts poorer countries more

| | Low-income AUC | High-income AUC |
|---|---|---|
| Country seen in training | 0.823 | 0.862 |
| Country held out | 0.739 | 0.819 |
| Cost of being unseen | 0.084 | 0.043 |

(Gradient-boosted trees; Presto + trees: 0.073 and 0.035.)

### 3.3 Not a labelling artefact

Low-income countries draw a median 30% of their labels from Geo-Wiki,
against 92% for high-income countries. Restricting scoring to Geo-Wiki points
(56 countries) removes that difference and widens the gap to 0.159
(0.107–0.211, trees) and 0.171 (0.117–0.231, Presto).

### 3.4 One-sided errors and area estimates

With the default 0.5 threshold, unseen low-income countries get recall 0.56
and specificity well above it, while high-income countries get recall 0.91.
For an area estimate by classify-and-count, that is −6.6 percentage points of
cropland share in low-income and +9.8 in high-income countries.

### 3.5 A deployed map

ESA WorldCover (class 40 = cropland) read at each point agrees with
CropHarvest's crop label at 42–46% recall in every income group. On Geo-Wiki
labels alone, its balanced accuracy is 0.072 (0.019–0.121) lower in low-income
countries.

### 3.6 What predicts failure

OLS on per-country AUC with robust errors (n = 66): crop share (+0.046 per SD,
p = 0.003) and log distance to the nearest labelled point in another country
(−0.045 per SD, p = 0.021) are significant; the low-income indicator falls from
−0.055 (p = 0.08) alone to −0.037 (p = 0.39) with controls. R² = 0.37.

### 3.7 Trust score

Features: mean |p − 0.5| on the country's own (unlabelled) points, the spread of
predictions, and the mean measured AUC of the k = 5 nearest countries in
standardised Presto-embedding centroid space. Linear model, validated
leave-one-country-out: ρ = 0.72, MAE 0.061 (constant: 0.085). Of the 10 worst
countries, 3 are among the 10 the score flags (trees; 4 for Presto).

### 3.8 Fixes

- Country-balanced sample weights (power 0.5): balanced accuracy 0.673 → 0.690
  overall and 0.659 → 0.668 in low-income countries; AUC unchanged.
- Platt recalibration on the 8 most similar countries (feature or geographic
  neighbours): no reliable gain.
- Saerens EM prior-shift: harmful (absolute share error 9.4 → 25.4 points),
  because unseen-country probabilities are not calibrated.
- 20 local labels for recalibration: absolute share error 9.4 → 7.6 points and
  low-income under-count −6.6 → −3.6.

## 4. Limitations

Point labels and mixed years; crowd labels may be noisier on smallholder
fields; dated income classes; 66 countries limit power; Prithvi without spatial
context; WorldCover's class definitions differ from CropHarvest's.

## 5. Next steps

1. Add field size (smallholder share) as an explanatory variable.
2. Run Prithvi on HLS image chips around each point, so it is used as designed.
3. Fine-tune Presto with country-balanced and group-robust objectives.
4. Turn the benchmark into an open leaderboard ranked by worst-country
   performance rather than the average.
