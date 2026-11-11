# Genie Agent: Airbnb prices

Created in Databricks under **Genie → New**. Name: *Airbnb prices: Berlin, Vienna, Munich*. Warehouse: the serverless starter warehouse (free).

## 1. Tables

- `workspace.airbnb.gold_listing_features`
- `workspace.airbnb.gold_city_metrics`
- `workspace.airbnb.gold_neighbourhood_metrics`
- `workspace.airbnb.gold_price_predictions`
- `workspace.airbnb.gold_model_metrics`

All tables and columns already have descriptions (written by the pipeline and the model notebooks). Genie reads them, so most of the meaning does not need repeating in the instructions. The instructions below cover what the descriptions cannot: the join, the filters people forget and the words users will type.

Under **Configure → Data**, for `gold_listing_features` turn on value sampling (entity matching) for `city`, `room_type` and `neighbourhood`. Then "Mitte" or "Leopoldstadt" or "entire homes" is matched to the real value.

## 2. General instructions (paste as is)

```
You answer questions about Airbnb listings in Berlin, Vienna and Munich and about a model that predicts their nightly price. All prices are in EUR per night.

Cities
- The column city is lower case: 'berlin', 'vienna', 'munich'. Users write "Berlin", "City A" or "Wien".
- City A = berlin, city B = vienna (the training cities). City C = munich (the new client city the model never saw during training).
- Each table can hold more than one snapshot of a city. Always use the latest snapshot_date of each city unless the user asks for a date.

Prices
- "Price" means price_eur in gold_listing_features. Many listings have no price (price_eur is NULL, price_missing = true). Never count them as 0.
- For price statistics on listings (median, average, ranges) use only rows where is_trainable = true. This leaves out listings without a price and extreme outliers. Say so in one short sentence in the answer.
- Use the median, not the average, when the user asks for a typical price, unless they say "average" or "mean". Median: percentile(price_eur, 0.5).
- For counts of listings ("how many listings", "most listings"), count all listings, with or without a price.
- Ready-made city and neighbourhood metrics are in gold_city_metrics and gold_neighbourhood_metrics. Prefer them for simple counts and medians. In gold_city_metrics, room_type = 'All' is the whole city.

Room types
- Values: 'Entire home/apt', 'Private room', 'Hotel room', 'Shared room'.
- "Entire home", "entire apartment", "whole flat" = 'Entire home/apt'.

Model and errors
- Predictions are in gold_price_predictions. Join to gold_listing_features on BOTH city and listing_id.
- gold_price_predictions has one row per listing and model_version. Always filter on one model_version before joining. If the user does not name a version, use the latest version that exists for that city and say which one you used.
- Error = predicted_price_eur - price_eur. Below zero means the model UNDER-prices the listing (predicts less than the real price). Above zero means it OVER-prices it.
- Errors only make sense on listings the model did not learn from:
  - berlin and vienna: is_trainable = true AND is_test = true (the held-out test set);
  - munich: is_trainable = true (every Munich listing is new to the model; ignore is_test).
- MAE = mean absolute error in EUR = avg(abs(predicted_price_eur - price_eur)).
- "Where does the model under-price most" means the average error (bias) per neighbourhood or room type, most negative first. Only show groups with at least 10 evaluated listings.
- Official model results (MAE of model and baseline per city, room type, stay type) are in gold_model_metrics. Use them when the user asks for "the MAE" or "how good is the model". evaluated_on = 'test' is Berlin and Vienna, 'new_city' is Munich. segment = 'all' is the whole city. The row city = 'berlin+vienna' is both training cities together.
- The baseline is the median train price for the same room type and number of guests.

Style
- Round prices and errors to whole EUR or 1 decimal.
- Do not use neighbourhood names to compare cities, they are different in every city.
```

## 3. Example SQL queries (Configure → Instructions → SQL queries)

Add each as a trusted example. Genie uses them as patterns for similar questions.

**What is the median price of entire homes in Berlin?**

```sql
SELECT round(percentile(price_eur, 0.5), 1) AS median_price_eur, count(*) AS priced_listings
FROM workspace.airbnb.gold_listing_features
WHERE city = 'berlin'
  AND room_type = 'Entire home/apt'
  AND is_trainable
  AND snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_listing_features WHERE city = 'berlin')
```

**Which neighbourhoods in Vienna have the most listings?**

```sql
SELECT neighbourhood, listings, round(median_price_eur, 1) AS median_price_eur
FROM workspace.airbnb.gold_neighbourhood_metrics
WHERE city = 'vienna'
  AND snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_neighbourhood_metrics WHERE city = 'vienna')
ORDER BY listings DESC
LIMIT 10
```

**Where does the model under-price listings the most?**

```sql
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
)
SELECT f.city, f.neighbourhood, count(*) AS n_listings,
       round(avg(p.predicted_price_eur - f.price_eur), 1) AS avg_error_eur,
       round(avg(abs(p.predicted_price_eur - f.price_eur)), 1) AS mae_eur
FROM workspace.airbnb.gold_listing_features f
JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
JOIN workspace.airbnb.gold_price_predictions p
  ON p.city = f.city AND p.listing_id = f.listing_id AND p.model_version = 'v1'
WHERE f.is_trainable
  AND (f.is_test OR f.city = 'munich')
GROUP BY f.city, f.neighbourhood
HAVING count(*) >= 10
ORDER BY avg_error_eur ASC
LIMIT 10
```

**What is the MAE of the model and the baseline in each city?**

```sql
SELECT model_version, city, evaluated_on, n_listings, mae_baseline_eur, mae_model_eur, improvement_pct, bias_eur
FROM workspace.airbnb.gold_model_metrics
WHERE segment = 'all'
ORDER BY model_version, evaluated_on, city
```

**Compare Berlin and Vienna**

```sql
SELECT city, listings, round(median_price_eur, 1) AS median_price_eur,
       round(median_availability_365) AS median_availability_365,
       round(avg_review_rating, 2) AS avg_review_rating, hosts,
       round(100 * share_long_stay, 1) AS pct_long_stay
FROM workspace.airbnb.gold_city_metrics m
WHERE room_type = 'All' AND city IN ('berlin', 'vienna')
  AND snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city)
```

## 4. Sample questions (shown to users on the start screen)

- What is the median price of entire homes in Berlin?
- Which neighbourhoods in Vienna have the most listings?
- Where does the model under-price listings the most?
- How much more expensive is Munich than Berlin and Vienna?
- How good is the model compared with the baseline?

## 5. Benchmarks (Configure → Benchmarks)

Add the five questions from section 3 with their SQL as the expected answer, then **Run benchmarks**. All five must pass before the demo. A question that fails is fixed by changing the general instructions, not by rewording the question. Screenshot the benchmark result for the presentation.

Extra checks to ask by hand (the answers must follow the rules above):

| Question | What a correct answer does |
|---|---|
| What is the average price in Munich? | uses `is_trainable`, says average, not median |
| How many listings does City A have? | counts all Berlin listings (12,376), not only priced ones |
| Which listing is the most over-priced in Vienna? | test rows only, error above zero, largest first |
| How wrong is the model in Munich? | `new_city` row of `gold_model_metrics`, MAE about 150 EUR, bias negative |

## 6. Tricks that made Genie answer correctly (for the presentation slide)

1. Column and table descriptions written by the pipeline: Genie reads them, so "is_trainable", "is_test" and "price_eur" are explained once, in the data, not in a long prompt.
2. One rule for each mistake Genie made in testing, written as a short instruction: lower-case city names, filtering on `model_version` before the join, the sign of the error, test rows only.
3. Trusted example SQL for the three required questions.
4. Value sampling on `city`, `room_type`, `neighbourhood`, so free text like "entire homes" or "Wien" maps to the stored values.
5. Benchmarks: the example questions with expected SQL, re-run after every change of the instructions.
