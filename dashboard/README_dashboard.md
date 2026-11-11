# AI/BI dashboard

Name in Databricks: **Airbnb prices: Berlin, Vienna, Munich**. Datasets: `datasets.sql` (one block per dataset, paste each into Data → Create from SQL and give it the name in its header). Every dataset reads only gold tables in `workspace.airbnb`.

Each page title states a finding (Lecture 7). The numbers in the titles below come from the pipeline and model READMEs. After building a page, check each title against what the page actually shows and correct the number if it differs.

The dashboard JSON is exported to `dashboard/airbnb_prices.lvdash.json` (File → Export) after it is finished, so it can be imported again.

## Global filter

One filter widget on every page: **City** (field `city`, all datasets). Default: all cities.

## Page 1. City overview

**Title:** *Munich is 60 percent more expensive than Berlin and Vienna, which cost almost the same*

| Widget | Dataset | Settings |
|---|---|---|
| 3 counters per city: listings, median price, hosts | `ds_city_overview` | one counter per value, filter on one city each, or a table with `city`, `listings`, `median_price_eur`, `hosts`, `listings_per_host` |
| Table: city profile | `ds_city_overview` | `city`, `snapshot_date`, `listings`, `median_price_eur`, `median_availability_365`, `avg_review_rating`, `pct_superhost`, `pct_long_stay`, `pct_without_price` |
| Bar: room-type mix | `ds_room_mix` | x = `city`, y = `pct_of_city_listings`, colour = `room_type` (stacked) |
| Bar: median price by room type | `ds_room_mix` | x = `room_type`, y = `median_price_eur`, colour = `city` (grouped) |
| Histogram: price | `ds_listings` | filter `is_trainable = true`, x = `price_eur` (bins of 20 EUR, max 600), colour = `city` |
| Histogram: availability | `ds_listings` | x = `availability_365` (bins of 30 days), colour = `city` |
| Bar: top 10 neighbourhoods by median price | `ds_neighbourhoods` | x = `median_price_eur`, y = `neighbourhood` (horizontal), sort desc, limit 10. Add the City filter so only one city is on screen |
| Bar: neighbourhoods with most listings | `ds_neighbourhoods` | same, sorted by `listings` |

## Page 2. Berlin against Vienna

**Title:** *Berlin and Vienna have the same price level, but a third of Berlin listings are long stays against one in ten in Vienna*

| Widget | Dataset | Settings |
|---|---|---|
| Table: side by side | `ds_compare_ab` | `metric`, `berlin`, `vienna`, `vienna_minus_berlin` |
| Bar: median price by room type | `ds_room_mix` | filter city in (Berlin, Vienna), x = `room_type`, y = `median_price_eur`, colour = `city` |
| Bar: median price by stay type | `ds_listings` | filter Berlin, Vienna and `is_trainable`, x = `stay_type`, y = MEDIAN(`price_eur`), colour = `city` |
| Line: median price by distance to centre | `ds_listings` | filter Berlin, Vienna and `is_trainable`, x = `distance_band`, y = MEDIAN(`price_eur`), colour = `city` |
| Text box | none | 2-3 sentences: what differs and why it matters for the model (long stays cost about half per night; Berlin has many more of them; the share of listings without price is higher in Berlin) |

The differences that matter for the model: the share of long-stay listings (`minimum_nights` is the strongest feature), the share of listings without a price (they cannot be trained on) and the price drop with distance from the centre.

## Page 3. Model

**Title:** *The model beats the baseline by 31 percent in Berlin and Vienna, but under-prices almost every Munich listing*

Put one filter on this page: **Model version** (field `model_version` of `ds_predictions`, `ds_model_metrics` and `ds_error_by_neighbourhood`), default `v1`. When participant 3 writes `v2`, it appears in the filter without editing the SQL.

| Widget | Dataset | Settings |
|---|---|---|
| Table: the three numbers | `ds_model_metrics` | filter `segment = all`; `city`, `evaluated_on`, `n_listings`, `mae_baseline_eur`, `mae_model_eur`, `improvement_pct`, `bias_eur` |
| Bar: MAE model vs baseline | `ds_model_metrics` | filter `segment = all`, x = `city`, y = `mae_baseline_eur` and `mae_model_eur` (grouped) |
| Scatter: predicted vs actual | `ds_predictions` | x = `actual_price_eur`, y = `predicted_price_eur`, colour = `city`. Axis max 600 so a few very expensive listings do not squash the plot. Points under the diagonal are under-priced |
| Bar: error by room type | `ds_predictions` | x = `room_type`, y = AVG(`abs_error_eur`), colour = `city` |
| Bar: error by stay type | `ds_predictions` | x = `stay_type`, y = AVG(`abs_error_eur`), colour = `city` |
| Bar: neighbourhoods where the model under-prices most | `ds_error_by_neighbourhood` | x = `bias_eur`, y = `neighbourhood`, sort asc, limit 10, City filter on |
| Table: most under-priced listings | `ds_predictions` | filter `direction = under-priced`, sort `error_eur` asc, limit 20: `city`, `neighbourhood`, `room_type`, `accommodates`, `actual_price_eur`, `predicted_price_eur`, `error_eur`, `listing_url` |
| Table: most over-priced listings | `ds_predictions` | same, `direction = over-priced`, sort `error_eur` desc |

Errors are shown only for listings the model did not train on: the test rows of Berlin and Vienna, and every trainable Munich listing. Under-priced means `predicted_price_eur < actual price`.

Sort the under-priced and over-priced tables by `error_eur` in EUR, not by `error_pct`. In percent, a listing at 20 EUR predicted at 60 EUR tops the list, which says little about real money lost.

### Munich on the Model page

The Munich rows are already in `ds_predictions` and `ds_model_metrics` (`evaluated_on = new city (never seen)`). With the City filter set to Munich, the page tells the city C story:

- MAE 149.61 EUR against 40.39 EUR on the Berlin and Vienna test set, and 163.24 EUR for the baseline on Munich;
- bias of about -143 EUR: the scatter lies almost entirely under the diagonal;
- long stays transfer (MAE about 45 EUR), short stays do not (about 152 EUR).

A text box under the Munich widgets explains the gap (see `README.md`, section City C).
