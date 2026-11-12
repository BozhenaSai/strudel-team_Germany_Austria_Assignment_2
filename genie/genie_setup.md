# Genie Agent: Airbnb prices

Created in Databricks under **Genie → New**. Name: *Airbnb prices: Berlin, Vienna, Munich*. Warehouse: the serverless starter warehouse (free).

## 1. Tables

- `workspace.airbnb.gold_listing_features`
- `workspace.airbnb.gold_baseline_predictions` (baseline price of every listing, written by `baseline/01_baseline`)
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
- Always filter gold_price_predictions on one model_version. v1 is the model of the first run. v2 is the model after the changes. Both are trained on Berlin and Vienna only. No model is trained on Munich; never say a model was retrained on Munich.
- If the user does not name a version: use v1 for Berlin and Vienna. For Munich show v1 and v2 side by side on the same held-out listings (see below). Always say which version you used.
- Error = predicted_price_eur - price_eur. "Under-priced" means predicted_price_eur minus price_eur is below zero. Above zero means over-priced.
- Errors only make sense on listings the model did not learn from:
  - berlin and vienna: is_trainable = true AND is_test = true (the held-out test set), for every version;
  - munich, v1: is_trainable = true (all priced listings; v1 used nothing from Munich);
  - munich, v2: is_trainable = true AND is_test = true ONLY (v2 takes Munich's median price from the other Munich listings).
- To compare v1 and v2 on Munich, use the same rows for both: munich, is_trainable = true AND is_test = true (1,019 listings). The baseline price of every listing is in gold_baseline_predictions (join on city and listing_id).
- MAE = mean absolute error in EUR = avg(abs(predicted_price_eur - price_eur)).
- "Where does the model under-price most" means the average error (bias) per neighbourhood or room type, most negative first. Only show groups with at least 10 evaluated listings.
- Official model results (MAE of model and baseline per city, room type, stay type) are in gold_model_metrics. Use them when the user asks for "the MAE" or "how good is the model". evaluated_on = 'test' is Berlin and Vienna, 'new_city' is Munich. segment = 'all' is the whole city. The row city = 'berlin+vienna' is both training cities together.
- In gold_model_metrics the Munich row of v1 covers all 5,076 priced listings and the Munich row of v2 only the 1,019 held-out listings. Do not compare these two rows directly; compute both on the held-out listings instead.
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

**Where does the model under-price listings the most?** (model v1, the first run)

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

**What is the MAE of the model and the baseline in each city?** (rows of different versions in Munich cover different listings, see the next example)

```sql
SELECT model_version, city, evaluated_on, n_listings, mae_baseline_eur, mae_model_eur, improvement_pct, bias_eur
FROM workspace.airbnb.gold_model_metrics
WHERE segment = 'all'
ORDER BY model_version, evaluated_on, city
```

**How do v1 and v2 compare in Munich?**

```sql
WITH m AS (
  SELECT listing_id, price_eur
  FROM workspace.airbnb.gold_listing_features
  WHERE city = 'munich' AND is_trainable AND is_test
    AND snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_listing_features WHERE city = 'munich')
)
SELECT 'baseline' AS predictor, count(*) AS n_listings, round(avg(abs(b.baseline_price_eur - m.price_eur)), 2) AS mae_eur
FROM m JOIN workspace.airbnb.gold_baseline_predictions b ON b.city = 'munich' AND b.listing_id = m.listing_id
UNION ALL
SELECT concat('model ', p.model_version), count(*), round(avg(abs(p.predicted_price_eur - m.price_eur)), 2)
FROM m JOIN workspace.airbnb.gold_price_predictions p ON p.city = 'munich' AND p.listing_id = m.listing_id
GROUP BY p.model_version
```

Expected: baseline 171.31, v1 156.90, v2 115.36 EUR on 1,019 listings.

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
- How do v1 and v2 compare in Munich?

## 5. Benchmarks (Configure → Benchmarks)

Add the six questions from section 3 with their SQL as the expected answer, then **Run benchmarks**. All six must pass before the demo. A question that fails is fixed by changing the general instructions, not by rewording the question. Screenshot the benchmark result for the presentation.

Extra checks to ask by hand (the answers must follow the rules above):

| Question | What a correct answer does |
|---|---|
| What is the average price in Munich? | uses `is_trainable`, says average, not median |
| How many listings does City A have? | counts all Berlin listings (12,376), not only priced ones |
| Which listing is the most over-priced in Vienna? | test rows only, error above zero, largest first |
| How wrong is the model in Munich? | v1: MAE 149.61 EUR on all priced listings, bias negative; and v1 against v2 on the same held-out listings (156.90 against 115.36 EUR) |
| Was the model retrained on Munich? | No. Both versions are trained on Berlin and Vienna only; v2 uses only Munich's median price |

## 6. Tricks that made Genie answer correctly (for the presentation slide)

1. Column and table descriptions written by the pipeline: Genie reads them, so "is_trainable", "is_test" and "price_eur" are explained once, in the data, not in a long prompt.
2. One rule for each mistake Genie made in testing, written as a short instruction: lower-case city names, filtering on one `model_version` before the join, the sign of the error, test rows only, and the same held-out rows when v1 and v2 are compared on Munich (Genie first compared 149.61 with 115.36, which cover different listings).
3. Trusted example SQL for the three required questions.
4. Value sampling on `city`, `room_type`, `neighbourhood`, so free text like "entire homes" or "Wien" maps to the stored values.
5. Benchmarks: the example questions with expected SQL, re-run after every change of the instructions.
