# Price model

Predicts the nightly price (`price_eur`) of a listing from `workspace.airbnb.gold_listing_features`.
Trained on Berlin and Vienna, then applied unchanged to a new city.

| Notebook | What it does | Parameters |
|---|---|---|
| `00_model_common` | Feature lists, preprocessing, baseline, metric, table writers. Pulled in with `%run`, not run on its own. | none |
| `01_train_evaluate` | Chooses the model by cross-validation on the train rows, fits it, reports test-set MAE against the baseline, saves the model, writes predictions for the training cities. | `train_cities` = `berlin,vienna`, `model_version` = `v1` |
| `02_score_city` | Loads the saved model and scores another city. Fits nothing. Reports the error on the new city next to the training cities and explains the gap. | `cities` = `munich`, `model_version` = `v1` |

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

Not run yet. `02_score_city` will apply model v1 unchanged after the `v1-AB` tag; the first-run result and the model changes go here.
