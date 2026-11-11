# Price model

Predicts the nightly price (`price_eur`) of a listing from `workspace.airbnb.gold_listing_features`.
Trained on Berlin and Vienna, then applied unchanged to a new city.

| Notebook | What it does | Parameters |
|---|---|---|
| `00_model_common` | Feature lists, preprocessing, baseline, metric, table writers. Pulled in with `%run`, not run on its own. | none |
| `01_train_evaluate` | Chooses the model by cross-validation on the train rows, fits it, reports test-set MAE against the baseline, saves the model, writes predictions for the training cities. | `train_cities` = `berlin,vienna`, `model_version` = `v1` |
| `02_score_city` | Loads the saved model and scores another city. Fits nothing. Reports the error on the new city next to the training cities and explains the gap. | `cities` = `munich`, `model_version` = `v1` |
| `03_model_v2` | The model changes after the first run on city C, applied one on top of the other and measured on Munich's held-out rows. Saves model v2 and writes its predictions for all three cities. | `new_city` = `munich`, `base_version` = `v1`, `new_version` = `v2` |

## Rules the model follows

- **Split.** The pipeline's `is_test` flag (about 20% per city, hash of the id). Train = `is_trainable` and not `is_test`. The model never makes its own split.
- **Model choice before the test set.** Candidates are compared by 5-fold cross-validation inside the train rows. The test rows are used once, for the reported numbers.
- **Target.** `log(price_eur)`, converted back with `exp`. The boosting model uses absolute-error loss, so it predicts the median price of similar listings, which is what MAE rewards.
- **Baseline.** Median `price_eur` by `room_type` and `accommodates` on the train rows; fallback to the room-type median, then the overall median.
- **No leakage.** Only columns from the pipeline's allowed list. Never `listing_id`, `snapshot_date`, `neighbourhood`, `price_missing`, `outlier_class`, `is_trainable`, `is_test`.
- **New city.** `is_test` is ignored; the error is measured on every `is_trainable` row.

## Candidates

| Candidate | Features |
|---|---|
| `ridge \| core` | linear reference on the portable features |
| `gbm \| core_no_avail` | portable features without `availability_365` (the EDA asked to test with and without it) |
| `gbm \| core` | portable features: size, room and property type, stay length, amenities, reviews, host, distance to centre |
| `gbm \| core_city` | + `city` |
| `gbm \| core_city_latlon` | + `city`, `latitude`, `longitude` |

Choice rule: lowest cross-validated MAE; a candidate earlier in the list wins if it is within 1% of the best.

Cross-validated MAE on the train rows (EUR):

| Candidate | Both cities | Berlin | Vienna |
|---|---|---|---|
| baseline | 59.75 | 65.04 | 55.38 |
| `ridge \| core` | 60.19 | 54.69 | 64.72 |
| `gbm \| core_no_avail` | 44.43 | 48.87 | 40.77 |
| `gbm \| core` | 43.84 | 48.07 | 40.35 |
| `gbm \| core_city` | 41.91 | 45.53 | 38.92 |
| **`gbm \| core_city_latlon`** (chosen) | **41.11** | 44.95 | 37.95 |

**Automatic feature selection** (step 2b in `01_train_evaluate`) is available but switched off for v1 (`AUTO_SELECT_FEATURES = False`): v1 uses all 31 features of the chosen set. It removes features by backward elimination on the train rows and keeps the smallest list whose cross-validated MAE is within 1% of the best.

## Outputs

| Output | Content |
|---|---|
| `gold_price_predictions` | `city`, `listing_id`, `predicted_price_eur`, `model_version`: every listing, also those without a price. Filter on one `model_version` before joining to `gold_listing_features` on `(city, listing_id)`. |
| `gold_model_metrics` | MAE of model and baseline, bias and median error per city, room type and stay type. `evaluated_on` = `test` or `new_city`. |
| `/Volumes/workspace/airbnb/models/price_model_<version>.joblib` | The fitted model with its feature list, known property types, baseline medians and training ranges. |

## Results (test set)

Model v1: `gbm | core_city_latlon`, gradient boosting on log price with 31 features. Trained on 14,294 listings (2026-11-10 20:33 UTC), tested on 3,673 held-out listings.

| | Baseline MAE (EUR) | Model MAE (EUR) | Improvement | Test listings |
|---|---|---|---|---|
| Berlin | 63.80 | 42.85 | 32.8% | 1,675 |
| Vienna | 54.85 | 38.33 | 30.1% | 1,998 |
| Berlin + Vienna | 58.93 | 40.39 | 31.5% | 3,673 |

- The typical listing is off by 21.61 EUR (median absolute error); a few expensive listings pull the MAE up. Listings for 7 or more guests have an MAE of 95.78 EUR.
- The model under-prices by 13.73 EUR on average. It predicts the median price of similar listings, and the median of a right-skewed price is below its mean.
- Most important features (MAE increase in EUR when shuffled): `minimum_nights` 16.70, `accommodates` 8.68, `bedrooms` 6.18, `latitude` 3.91, `dist_to_centre_km` 3.12, `longitude` 2.75.

Saved model: `/Volumes/workspace/airbnb/models/price_model_v1.joblib`
sha256: `e5f7f5b86262e63915c9f9d5f7d9c1ed2920517d9baa2adf35de8f5e092bf827`

## City C (Munich)

Snapshot 2026-09-23. Pipeline and model as tagged `v1-AB`.

### First run: model v1, unchanged

It ran without errors. The saved model was the same file (same sha256).

| | MAE (EUR) | Listings |
|---|---|---|
| Baseline on Munich (Berlin + Vienna medians) | 163.24 | 5,076 |
| Model on Berlin + Vienna (test) | 40.39 | 3,673 |
| Model on Munich, first run | 149.61 | 5,076 |

What the first run showed:

- **The model under-prices almost every listing.** Bias −142.97 EUR, nearly equal to the MAE.
- **Features the model had never seen.** `city = munich` is new for 100% of listings; every Munich longitude (11.39 to 11.72) is outside the training range (13.12 to 16.54), and 32% of latitudes are too.
- **A price level the model had never seen.** Median price 204 EUR against 126.50 EUR in training. The model's median prediction was 121.40 EUR.
- **Level is only part of it.** Rescaling all predictions by one factor (1.68) lowers the MAE to 112.83 EUR, a 24.6% cut. Munich prices are also spread wider: median 155 EUR for 1 to 2 guests and 526 EUR for 7 or more; 293 EUR within 2 km of the centre and 125 EUR beyond 10 km.
- **The ranking mostly holds.** Rank correlation of predicted and actual price: 0.67.
- **Long stays transferred, short stays did not.** MAE 45.37 EUR for long stays (125 listings), 152.24 EUR for short stays (4,951 listings).
- **Assumption, not verified:** the snapshot date falls during Oktoberfest (19 September to 4 October 2026), which may inflate short-stay prices.

### Changes

**Pipeline fixes:** none. Participant 2 checked the Munich run (encoding, types, outliers, missing values, row balance, city centre, feature distributions) and found no defect. A rerun with the same model gives the same MAE.

**Model changes** (`03_model_v2`):

| # | Change | Reason | Munich prices used |
|---|---|---|---|
| A | Remove `city`, `latitude`, `longitude` | Unseen city value; coordinates outside the training range. `dist_to_centre_km` carries location the same way in every city. | none |
| B | Predict price relative to the city's median price | A price level the model never saw | one number: the median of Munich's train rows |
| C | Retrain on Berlin + Vienna + Munich train rows (= v2) | One factor is not enough: Munich prices are spread wider | Munich's train rows |

### After the changes

Every row is measured on the same 1,019 held-out Munich listings (`is_test = true`). On these rows the first run gives 156.90 EUR (149.61 EUR on all 5,076 priced listings).

| Step | MAE on Munich (EUR) | MAE on Berlin + Vienna test (EUR) |
|---|---|---|
| Baseline, Berlin + Vienna medians | 171.31 | 58.93 |
| Baseline, Munich medians | 122.93 | n/a |
| v1, first run | 156.90 | 40.39 |
| A: without city, latitude, longitude | 149.18 | 42.87 |
| B: A + price relative to the city level | 115.36 | 42.94 |
| C: B retrained with Munich = v2 | 106.53 | 43.50 |

Model v2 per city (test rows):

| City | Baseline MAE (EUR) | Model v2 MAE (EUR) | Improvement | Bias (EUR) | Median price (EUR) |
|---|---|---|---|---|---|
| Berlin | 63.80 | 46.37 | 27.3% | −27.01 | 128.75 |
| Vienna | 54.85 | 41.09 | 25.1% | −2.32 | 124.50 |
| Munich | 122.93 | 106.53 | 13.3% | −49.50 | 220.00 |

### The gap, explained

- **Features that did not transfer: about 8 EUR.** Removing `city` and the raw coordinates lowers the Munich MAE from 156.90 to 149.18 EUR. It costs about 2.5 EUR in Berlin and Vienna, where coordinates were useful.
- **Price level: about 34 EUR, the largest step.** One number, Munich's median price (201 EUR against 129 in Berlin and 125 in Vienna), lowers the MAE to 115.36 EUR.
- **Munich's own price structure: about 9 EUR.** Retraining with Munich's train rows lowers it to 106.53 EUR. The ranking improves from 0.67 to 0.73 (rank correlation).
- **What remains is not a transfer problem.** After retraining, the Munich error is still 2.4 times the error in Berlin and Vienna. Munich's own baseline shows the same: 122.93 EUR against 58.93 EUR at home. Prices are higher, so the same relative error is more euros, and listings that look alike differ much more in price. The typical listing is off by 61.01 EUR; a few expensive ones pull the MAE up to 106.53 EUR.
- **Long stays are predicted well** (MAE 16.88 EUR, 23 listings); short stays carry the error (108.60 EUR, 996 listings).
- **Assumption, not verified:** Oktoberfest pricing during the snapshot would explain both the high level and the wide spread of short-stay prices. Model v2 has learned these prices as normal for Munich.
- **Cost of the portable model:** v2 is 3.11 EUR worse than v1 on the Berlin + Vienna test set (43.50 against 40.39 EUR).

Saved model v2: `/Volumes/workspace/airbnb/models/price_model_v2.joblib`
sha256: `bce2082ed8f0002f668d87b0c6ea74c147c2885b4519184d8bca83356a675985`