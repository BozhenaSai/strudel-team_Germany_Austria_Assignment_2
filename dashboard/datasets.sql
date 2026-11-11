-- AI/BI dashboard "Airbnb prices: Berlin, Vienna, Munich"
-- One block per dataset. In the dashboard editor: Data -> Create from SQL, paste the block, rename the dataset
-- to the name in the header. Every dataset reads only gold tables in workspace.airbnb.
-- Every dataset keeps the latest snapshot of each city, so a re-run with a newer file does not double the numbers.


-- =====================================================================================================
-- ds_city_overview   (page 1 and 2: one row per city, whole-city metrics)
-- =====================================================================================================
SELECT
  initcap(m.city)                                   AS city,
  m.snapshot_date,
  m.listings,
  m.listings_with_price,
  round(m.median_price_eur, 1)                      AS median_price_eur,
  round(m.median_availability_365, 0)               AS median_availability_365,
  round(m.avg_review_rating, 2)                     AS avg_review_rating,
  m.hosts,
  round(m.listings / m.hosts, 2)                    AS listings_per_host,
  round(100 * m.share_long_stay, 1)                 AS pct_long_stay,
  round(100 * m.share_superhost, 1)                 AS pct_superhost,
  round(100 * (1 - m.listings_with_price / m.listings), 1) AS pct_without_price
FROM workspace.airbnb.gold_city_metrics m
WHERE m.room_type = 'All'
  AND m.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city);


-- =====================================================================================================
-- ds_room_mix   (page 1 and 2: room type mix and median price per room type)
-- =====================================================================================================
SELECT
  initcap(m.city)                      AS city,
  m.room_type,
  m.listings,
  round(100 * m.share_of_listings, 1)  AS pct_of_city_listings,
  round(m.median_price_eur, 1)         AS median_price_eur
FROM workspace.airbnb.gold_city_metrics m
WHERE m.room_type <> 'All'
  AND m.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city);


-- =====================================================================================================
-- ds_neighbourhoods   (page 1: neighbourhoods by price; only districts with 20+ priced listings,
--                      so a median from 3 listings does not top the chart)
-- =====================================================================================================
SELECT
  initcap(n.city)                        AS city,
  n.neighbourhood,
  n.listings,
  n.listings_with_price,
  round(n.median_price_eur, 1)           AS median_price_eur,
  round(n.avg_review_rating, 2)          AS avg_review_rating,
  round(n.avg_availability_365, 0)       AS avg_availability_365,
  round(100 * n.share_entire_home, 1)    AS pct_entire_home
FROM workspace.airbnb.gold_neighbourhood_metrics n
WHERE n.listings_with_price >= 20
  AND n.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_neighbourhood_metrics x WHERE x.city = n.city);


-- =====================================================================================================
-- ds_listings   (page 1 and 2: price and availability distributions, price vs distance to centre)
--               only trainable rows for price charts: no missing prices, no outliers
-- =====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
)
SELECT
  initcap(f.city)                 AS city,
  f.listing_id,
  f.neighbourhood,
  f.room_type,
  f.accommodates,
  CASE WHEN f.accommodates <= 2 THEN '1-2'
       WHEN f.accommodates <= 4 THEN '3-4'
       WHEN f.accommodates <= 6 THEN '5-6'
       ELSE '7+' END              AS guests_group,
  CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
  f.price_eur,
  f.availability_365,
  round(f.dist_to_centre_km, 2)   AS dist_to_centre_km,
  CASE WHEN f.dist_to_centre_km < 2  THEN 'a) 0-2 km'
       WHEN f.dist_to_centre_km < 5  THEN 'b) 2-5 km'
       WHEN f.dist_to_centre_km < 10 THEN 'c) 5-10 km'
       ELSE 'd) 10+ km' END       AS distance_band,
  f.review_scores_rating,
  f.has_reviews,
  f.host_is_superhost,
  f.is_trainable,
  f.outlier_class
FROM workspace.airbnb.gold_listing_features f
JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date;


-- =====================================================================================================
-- ds_compare_ab   (page 2: Berlin against Vienna, one row per metric, with the difference)
-- =====================================================================================================
WITH c AS (
  SELECT m.city, CAST(m.listings AS DOUBLE) AS listings,   -- stack() needs one type per column
         m.median_price_eur, m.median_availability_365, m.avg_review_rating,
         m.listings / m.hosts AS listings_per_host, m.share_long_stay, m.share_superhost,
         1 - m.listings_with_price / m.listings AS share_without_price
  FROM workspace.airbnb.gold_city_metrics m
  WHERE m.room_type = 'All' AND m.city IN ('berlin', 'vienna')
    AND m.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city)
),
p AS (
  SELECT
    max(CASE WHEN city = 'berlin' THEN listings END)                AS b_listings,
    max(CASE WHEN city = 'vienna' THEN listings END)                AS v_listings,
    max(CASE WHEN city = 'berlin' THEN median_price_eur END)        AS b_price,
    max(CASE WHEN city = 'vienna' THEN median_price_eur END)        AS v_price,
    max(CASE WHEN city = 'berlin' THEN median_availability_365 END) AS b_avail,
    max(CASE WHEN city = 'vienna' THEN median_availability_365 END) AS v_avail,
    max(CASE WHEN city = 'berlin' THEN avg_review_rating END)       AS b_rating,
    max(CASE WHEN city = 'vienna' THEN avg_review_rating END)       AS v_rating,
    max(CASE WHEN city = 'berlin' THEN listings_per_host END)       AS b_lph,
    max(CASE WHEN city = 'vienna' THEN listings_per_host END)       AS v_lph,
    max(CASE WHEN city = 'berlin' THEN share_long_stay END)         AS b_long,
    max(CASE WHEN city = 'vienna' THEN share_long_stay END)         AS v_long,
    max(CASE WHEN city = 'berlin' THEN share_superhost END)         AS b_super,
    max(CASE WHEN city = 'vienna' THEN share_superhost END)         AS v_super,
    max(CASE WHEN city = 'berlin' THEN share_without_price END)     AS b_noprice,
    max(CASE WHEN city = 'vienna' THEN share_without_price END)     AS v_noprice
  FROM c
)
SELECT metric, round(berlin, 2) AS berlin, round(vienna, 2) AS vienna, round(vienna - berlin, 2) AS vienna_minus_berlin
FROM p
LATERAL VIEW stack(8,
  '1 Listings',                         b_listings,     v_listings,
  '2 Median price (EUR)',               b_price,        v_price,
  '3 Median availability (days/year)',  b_avail,        v_avail,
  '4 Avg review score (0-5)',           b_rating,       v_rating,
  '5 Listings per host',                b_lph,          v_lph,
  '6 Long-stay listings (%)',           100 * b_long,   100 * v_long,
  '7 Superhosts (%)',                   100 * b_super,  100 * v_super,
  '8 Listings without price (%)',       100 * b_noprice, 100 * v_noprice
) s AS metric, berlin, vienna
ORDER BY metric;


-- =====================================================================================================
-- ds_model_metrics   (page 3: MAE of model and baseline from the model notebooks)
--   evaluated_on = 'test'     -> held-out 20 % of Berlin / Vienna
--   evaluated_on = 'new_city' -> all priced Munich listings
-- =====================================================================================================
SELECT
  model_version,
  CASE WHEN city = 'berlin+vienna' THEN 'Berlin + Vienna' ELSE initcap(city) END AS city,
  CASE WHEN evaluated_on = 'test' THEN 'test set (training city)' ELSE 'new city (never seen)' END AS evaluated_on,
  segment,
  n_listings,
  mae_baseline_eur,
  mae_model_eur,
  improvement_pct,
  median_ae_eur,
  bias_eur,
  median_price_eur,
  evaluated_at
FROM workspace.airbnb.gold_model_metrics;


-- =====================================================================================================
-- ds_predictions   (page 3: one row per evaluated listing, predicted vs actual, errors)
--   Berlin and Vienna: only test rows (the model has seen the train rows, their error means nothing)
--   Munich: every trainable row (the whole city is unseen)
--   model_version is a column: put a filter on it in the dashboard, so v2 shows up without editing SQL
-- =====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
f AS (
  SELECT f.*,
         CASE WHEN f.city IN ('berlin', 'vienna') AND f.is_test THEN 'test set (training city)'
              WHEN f.city NOT IN ('berlin', 'vienna')          THEN 'new city (never seen)' END AS evaluated_on
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  WHERE f.is_trainable
)
SELECT
  p.model_version,
  initcap(f.city)                                     AS city,
  f.evaluated_on,
  f.listing_id,
  f.neighbourhood,
  f.room_type,
  f.accommodates,
  CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
  round(f.dist_to_centre_km, 2)                       AS dist_to_centre_km,
  f.price_eur                                         AS actual_price_eur,
  p.predicted_price_eur,
  round(p.predicted_price_eur - f.price_eur, 2)       AS error_eur,         -- below 0 = model under-prices
  round(abs(p.predicted_price_eur - f.price_eur), 2)  AS abs_error_eur,
  round(100 * (p.predicted_price_eur - f.price_eur) / f.price_eur, 1) AS error_pct,
  CASE WHEN p.predicted_price_eur < f.price_eur THEN 'under-priced' ELSE 'over-priced' END AS direction,
  concat('https://www.airbnb.com/rooms/', f.listing_id) AS listing_url
FROM f
JOIN workspace.airbnb.gold_price_predictions p
  ON p.city = f.city AND p.listing_id = f.listing_id
WHERE f.evaluated_on IS NOT NULL;


-- =====================================================================================================
-- ds_error_by_neighbourhood   (page 3: where the model is wrong; districts with 10+ evaluated listings)
-- =====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
e AS (
  SELECT p.model_version, f.city, f.neighbourhood, f.price_eur, p.predicted_price_eur
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable
    AND (f.city NOT IN ('berlin', 'vienna') OR f.is_test)
)
SELECT
  model_version,
  initcap(city)                                         AS city,
  neighbourhood,
  count(*)                                              AS n_listings,
  round(percentile(price_eur, 0.5), 1)                  AS median_actual_eur,
  round(percentile(predicted_price_eur, 0.5), 1)        AS median_predicted_eur,
  round(avg(abs(predicted_price_eur - price_eur)), 2)   AS mae_eur,
  round(avg(predicted_price_eur - price_eur), 2)        AS bias_eur          -- below 0 = under-priced on average
FROM e
GROUP BY model_version, city, neighbourhood
HAVING count(*) >= 10;
