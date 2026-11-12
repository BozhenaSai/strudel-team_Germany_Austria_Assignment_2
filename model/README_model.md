# Price model

Predicts the nightly price (`price_eur`) of a listing from `workspace.airbnb.gold_listing_features`.
Trained on Berlin and Vienna, then applied unchanged to a new city.

| Notebook | What it does | Parameters |
|---|---|---|
| `00_model_common` | Feature lists, preprocessing, baseline, metric, table writers. Pulled in with `%run`, not run on its own. | none |
| `01_train_evaluate` | Chooses the model by cross-validation on the train rows, fits it, reports test-set MAE against the baseline, saves the model, writes predictions for the training cities. | `train_cities` = `berlin,vienna`, `model_version` = `v1` |
| `02_score_city` | Loads the saved model and scores another city. Fits nothing. Reports the error on the new city next to the training cities and explains the gap. | `cities` = `munich`, `model_version` = `v1` |
| `03_model_v2` | The model changes after the first run on city C, applied one on top of the other and measured on Munich's held-out rows. Saves the chosen step as model v2 and writes its predictions for all three cities. | `new_city` = `munich`, `base_version` = `v1`, `new_version` = `v2`, `final_step` = `B` |

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

**Model v2 is step B.** The assignment reports the error on A and B and on city C separately, also after the changes, so Munich stays a city the model was not trained on. In step B the model is fitted on Berlin and Vienna rows only; Munich contributes one number, its median price, used when a prediction is converted to euros. No model is trained on Munich listings.

### After the changes

Every row is measured on the same 1,019 held-out Munich listings (`is_test = true`), because step B takes Munich's median price from the other Munich rows. On these listings the first run gives 156.90 EUR (149.61 EUR on all 5,076 priced listings).

| Step | Munich prices used | MAE on Munich (EUR) | MAE on Berlin + Vienna test (EUR) |
|---|---|---|---|
| Baseline, Berlin + Vienna medians | none | 171.31 | 58.93 |
| v1, first run | none | 156.90 | 40.39 |
| A: without city, latitude, longitude | none | 149.18 | 42.87 |
| **B: A + price relative to the city level = v2** | one number: Munich's median price | **115.36** | **42.94** |
| Reference: baseline from Munich's own medians (not a model, not used for predictions) | medians of Munich's `is_test = false` rows | 122.93 | n/a |

The three numbers, side by side:

| | MAE (EUR) |
|---|---|
| Baseline on Munich | 163.24 (all priced listings), 171.31 (held-out listings) |
| Model on Berlin + Vienna | 40.39 (v1), 42.94 (v2) |
| Model on Munich, first run | 149.61 (all priced listings), 156.90 (held-out listings) |
| Model on Munich, after the changes (v2) | 115.36 (held-out listings) |

### The gap, explained

- **Features that did not transfer: about 8 EUR.** Removing `city` and the raw coordinates lowers the Munich MAE from 156.90 to 149.18 EUR. It costs about 2.5 EUR in Berlin and Vienna, where coordinates were useful.
- **Price level: about 34 EUR, the largest step.** One number, Munich's median price (201 EUR against 129 in Berlin and 125 in Vienna), lowers the MAE to 115.36 EUR.
- **After the changes Munich is still 2.7 times worse than Berlin and Vienna** (115.36 against 42.94 EUR). The model was not trained on Munich, and a single price level does not capture that Munich prices are spread wider: size and a central location cost more there.
- **Munich prices are hard to predict even with Munich data.** A baseline built from Munich's own medians has an MAE of 122.93 EUR, against 58.93 EUR for the same baseline at home. Prices are higher, so the same relative error is more euros, and listings that look alike differ much more in price.
- **By neighbourhood, the error follows the price of the district** (participant 4, dashboard; 1,019 held-out listings, districts with 10 or more listings). v1 is worst in the most expensive districts: Altstadt-Lehel (median price 378 EUR, MAE 394 EUR, bias −391 EUR), Schwanthalerhöhe (254 EUR, MAE 220 EUR) and Schwabing-Freimann (271 EUR, MAE 190 EUR). It is best in cheaper outer districts such as Aubing-Lochhausen-Langwied (126 EUR, MAE 71 EUR) and Ramersdorf-Perlach (176 EUR, MAE 74 EUR). v1 predicts roughly the same Berlin and Vienna price level everywhere, so its bias grows with the district's price.
- **v2 fixes the outer districts, not the centre.** v2 is better than v1 and the baseline in every district. In outer districts its bias is close to zero (Trudering-Riem −2 EUR, Moosach −5 EUR, Aubing +2 EUR). In the expensive centre it still under-prices strongly (Altstadt-Lehel −266 EUR, Schwanthalerhöhe −104 EUR, Schwabing-Freimann −97 EUR). The premium for a central location is larger in Munich than in Berlin and Vienna. With 15 to 59 listings per district, these numbers are rough.
- **Long stays transferred, short stays carry the error** (first run: 45.37 against 152.24 EUR).
- **Assumption, not verified:** Oktoberfest pricing during the snapshot (23 September) would explain both the high level and the wide spread of short-stay prices.
- **Cost of the portable model:** v2 is 2.55 EUR worse than v1 on the Berlin + Vienna test set (42.94 against 40.39 EUR).

Model v2 on the held-out listings:

| City | Evaluated on | Listings | Baseline MAE (EUR) | Model v2 MAE (EUR) | Improvement |
|---|---|---|---|---|---|
| Berlin + Vienna | test | 3,673 | 58.93 | 42.94 | 27.1% |
| Munich | held-out listings, not trained on | 1,019 | 171.31 | 115.36 | 32.7% |

Model v2 on Munich in detail (1,019 held-out listings):

- **Still too low on average.** Bias −57.57 EUR (first run: −142.97 EUR on all priced listings). The typical short-stay listing is off by 69.23 EUR.
- **Ranking unchanged.** Rank correlation of predicted and actual price 0.68 (first run 0.67): the changes fixed the level, not the order.
- **By room type.** Entire homes: MAE 133.98 EUR against a baseline of 202.11 EUR. Private rooms: 82.30 against 118.35 EUR.
- **By size.** MAE grows with the number of guests: 84.56 EUR for 1 to 2 guests, 133.63 for 3 to 4, 218.00 for 5 to 6, 206.24 for 7 or more.
- **One price level does not fit every segment.** Short stays are still under-priced (bias −59.29 EUR), while long stays (23 listings, bias +16.86 EUR) and hotel rooms (9 listings, bias +58.16 EUR, worse than the baseline) are now over-priced. Munich's premium sits in short-stay homes and rooms, not in all listings.

Saved model v2: `/Volumes/workspace/airbnb/models/price_model_v2.joblib`
sha256: `2aaaa2327ae2f1fe972825540ce3754f20b409d2725c30ddf348b42817a6b2ef`