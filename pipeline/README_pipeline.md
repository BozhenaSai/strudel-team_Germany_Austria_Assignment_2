# ETL pipeline for Inside Airbnb listings

One Databricks Job turns a raw `listings.csv.gz` into cleaned listings, model features and dashboard metrics. The city is a parameter. To add a city, upload its file to the Volume and run the Job with new parameters. No code is copied.

## Layers

The raw file is read from `/Volumes/workspace/airbnb/raw/<city>/listings.csv.gz`. Every table lives in the schema `workspace.airbnb`, and the layer is the table prefix.

| Step | Notebook | Output |
|---|---|---|
| 1. Bronze | `01_bronze` | `bronze_listings` |
| 2. Silver | `02_silver` | `silver_listings`, `quarantine_listings` |
| 3. Gold | `03_gold` | `gold_listing_features`, `gold_city_metrics`, `gold_neighbourhood_metrics`, `gold_city_centre` |
| 4. Checks | `04_checks` | fails the Job if the output cannot be trusted |

All notebooks share `00_config`, which holds the paths, the parameters and the helper functions. Every stage also writes one line to `pipeline_run_log`: what came in, what came out and what was set aside.

## Parameters

| Parameter | Berlin (A) | Vienna (B) | Munich (C) |
|---|---|---|---|
| `city` | `berlin` | `vienna` | `munich` |
| `snapshot_date` | `2026-09-19` | `2026-09-17` | date of the Munich snapshot |
| `currency` | `EUR` | `EUR` | `EUR` |

The Job is defined in `job_definition.yml` and is called `airbnb_price_pipeline`. To run a city, open the Job and choose Run now with different settings. To run a single notebook by hand, set the same three values in its widgets.

Running a city again replaces only that city's rows (`replaceWhere`). It never duplicates data and never touches other cities.

## What each stage does

**Bronze.** Reads the file with `multiLine`, because the host text contains line breaks. Keeps all 90 source columns as text and adds `_city`, `_snapshot_date`, `_ingested_at` and `_source`. A column that changes shape in a new city cannot break the load.

**Silver.** One row is one listing on the snapshot date. It applies the cleaning rules from the EDA, each with its reason (see the table at the top of the notebook). Columns are selected by name, and a missing column becomes NULL instead of failing the job. Every cast is a `try_cast`, because serverless compute runs with ANSI mode on, and a plain cast of a bad string would raise an error.

- Quarantined rows go to `quarantine_listings` with the reason in `_failed_rules`. The rules are: no id, impossible coordinates, no room type, fewer than one guest.
- Listings without a price and price outliers are not quarantined. They are real listings, so they stay in silver with flags and are only left out of training.
- `outlier_class` is one of `no_price`, `implausible_low` (below 15 EUR), `extreme_high`, `extreme_low` (beyond 3 IQR of log price within city and room type) or `ok`. The rule is relative, so it needs no tuning for a city with a different price level.
- The row count is checked on every run: rows received equal silver plus quarantined plus duplicates removed.

**Gold.** The feature table has no leakage: no `estimated_*`, no `price_quote_*`, no `availability_30/60/90`, no ids and no free text. Missing `beds` and `bedrooms` are filled with the city median for the same number of guests. Missing review scores are filled with the city median, and `has_reviews` marks those listings. Location is `dist_to_centre_km`, the distance to the median coordinate of the city's listings (stored in `gold_city_centre`). This needs no per-city setup, and district names are not used because they do not carry over between cities. `is_test` marks about 20 percent of the listings in each city, chosen by a hash of the id, so it is the same on every run. Tables and columns have descriptions, which Genie and the dashboard read.

**Checks.** Silver has one row per listing. Gold has one row per silver listing. No leakage column is in `gold_listing_features`. No key is NULL. `is_trainable` never includes a listing without a price. The silver row count balances.

**Run log.** If a stage crashes, the error is written to `pipeline_run_log` before the Job fails. A first run on a new city therefore leaves a record of what broke.

## Results for the training cities

Both cities were run and the output matches the numbers below, which were calculated separately from the raw files.

| | Berlin | Vienna |
|---|---|---|
| rows in bronze and in silver | 12,376 | 12,982 |
| quarantined and duplicates | 0 and 0 | 0 and 0 |
| `ok` (equals `is_trainable`) | 8,138 | 9,829 |
| `no_price` | 4,219 | 3,113 |
| `extreme_high` and `implausible_low` | 16 and 3 | 38 and 2 |
| long-stay listings | 3,999 | 1,424 |
| city centre (lat, lon) | 52.5096, 13.4084 | 48.2038, 16.3598 |

## Not covered

- `calendar.csv.gz` and `reviews.csv.gz` are optional in the assignment and are not loaded.
- The pipeline is not tuned for city C. The first run on Munich uses the pipeline exactly as tagged `v1-AB`. Whatever breaks is documented and fixed afterwards as a pipeline fix.