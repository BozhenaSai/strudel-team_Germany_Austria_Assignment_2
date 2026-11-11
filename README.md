# Airbnb price lakehouse: Germany & Austria (team Strudel)

Group Assignment 2. A lakehouse over Inside Airbnb data on Databricks Free Edition, a model that suggests a nightly price for a listing, an AI/BI dashboard and a Genie Agent. The model is trained on two cities and then run, unchanged, on a third city it has never seen.

Region: **Germany & Austria**, currency **EUR** (no conversion needed).

## Cities and snapshots

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/) (CC BY 4.0), detailed `listings.csv.gz`, latest snapshot of each city.

| Role | City | Snapshot date | Listings |
|---|---|---|---|
| A (training) | Berlin | 2026-09-19 | 12,376 |
| B (training) | Vienna | 2026-09-17 | 12,982 |
| C (new client) | Munich | 2026-09-23 | 7,493 |

Why Munich as city C: it is in the same region and currency, but it is a much more expensive market than Berlin and Vienna, and a model that only works on the price level it was trained on should fail visibly there.

Munich was not opened until the pipeline and the model for Berlin and Vienna were finished and tagged `v1-AB`.

Raw files live in a Databricks Volume, not in git: `/Volumes/workspace/airbnb/raw/<city>/listings.csv.gz`.

## Repository

| Folder | Content | Details |
|---|---|---|
| `eda/` | `eda_berlin_vienna.ipynb`: exploratory analysis of Berlin and Vienna, cleaning rules, leakage review | |
| `pipeline/` | Databricks Job `airbnb_price_pipeline`: bronze → silver → gold → checks, city as a parameter | [`pipeline/README_pipeline.md`](pipeline/README_pipeline.md) |
| `baseline/` | `01_baseline.py`: baseline (median price by room type and guests) and its check against the model notebook | below |
| `model/` | training and evaluation (`01_train_evaluate`), scoring a new city (`02_score_city`) | [`model/README_model.md`](model/README_model.md) |
| `dashboard/` | SQL of every dashboard dataset, page layout and titles, exported dashboard | [`dashboard/README_dashboard.md`](dashboard/README_dashboard.md) |
| `genie/` | Genie Agent setup: tables, instructions, example SQL, benchmarks | [`genie/genie_setup.md`](genie/genie_setup.md) |
| `presentation/` | slides as PDF | |

All tables are in `workspace.airbnb`. The layer is the table prefix (`bronze_`, `silver_`, `gold_`).

## How to reproduce

1. Upload each city's `listings.csv.gz` to `/Volumes/workspace/airbnb/raw/<city>/`.
2. Run the Job `airbnb_price_pipeline` once per city (`city`, `snapshot_date`, `currency` as parameters, see the pipeline README).
3. Run `baseline/01_baseline` (check that the baseline matches the model notebook).
4. Run `model/01_train_evaluate` (`train_cities = berlin,vienna`, `model_version = v1`).
5. Run `model/02_score_city` (`cities = munich`, `model_version = v1`).
6. Open the dashboard and the Genie Agent. They read the gold tables, so nothing needs refreshing by hand.

## Baseline

The median `price_eur` of the train rows (Berlin and Vienna together, `is_trainable AND NOT is_test`) for the same `room_type` and `accommodates`. If that pair does not exist in train, the median of the room type is used, then the overall median.

The baseline was written twice: in pandas inside the model notebook and in Spark in `baseline/01_baseline.py`. The Spark notebook compares its MAE with `gold_model_metrics` and fails if any city differs by a cent or more. Both give the same numbers, so the model is compared against one agreed definition.

## Results

MAE in EUR. Berlin and Vienna: held-out test set (about 20 percent of listings per city). Munich: every priced listing.

| | Baseline | Model v1 |
|---|---|---|
| Berlin (test) | 63.80 | 42.85 |
| Vienna (test) | 54.85 | 38.33 |
| Berlin + Vienna (test) | 58.93 | 40.39 |
| Munich, first run, model unchanged | 163.24 | 149.61 |

## City C: Munich

**First run.** The pipeline ran without errors and all checks passed (one leftover test cell was removed before the run, see the pipeline README). The model ran too, but its error on Munich is almost four times the error on Berlin and Vienna, and it beats the baseline by only 8 percent instead of 31.

**Why the gap.** Most of the error is the price level, not the ranking of listings:

- Munich is much more expensive. The median price is 204 EUR against about 127 EUR in Berlin and Vienna. The model predicts prices it learned in Berlin and Vienna, so it under-prices almost every Munich listing (average error about -143 EUR).
- The model has never seen `city = munich`, and every Munich longitude lies outside the range of the training data. The location features therefore cannot tell the model that it is in an expensive city.
- Long-stay listings transferred well (MAE about 45 EUR), short stays did not (about 152 EUR). Only 3 percent of Munich listings are long stays, against 32 percent in Berlin, so the part of the market the model handles well is small in Munich.
- The snapshot falls in the Oktoberfest period, which may push short-stay prices up. This is an assumption we could not check with one snapshot.

The baseline fails for the same reason: it is also built from Berlin and Vienna prices.

**Changes.** Pipeline fixes are listed in the pipeline README, model changes (v2) in the model README. The Model page of the dashboard shows Munich next to Berlin and Vienna and switches between v1 and v2 with a filter.

## Team

| Participant | Part |
|---|---|
| 1 | data, EDA, leakage review, architecture, presentation |
| 2 | pipeline, team admin |
| 3 | model, city C model changes |
| 4 | baseline, dashboard, Genie Agent, repository |

## Presentation

`presentation/` (PDF). Link to be added.
