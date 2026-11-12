-- AI/BI dashboard "Airbnb prices: Berlin, Vienna, Munich": SQL of every dataset.
-- Generated together with airbnb_prices.lvdash.json (the dashboard itself; import it instead of rebuilding).
-- Every dataset reads only gold tables in workspace.airbnb and keeps the latest snapshot of each city.


-- ====================================================================================================
-- ds_city_overview   (One row per city: whole-city metrics.)
-- ====================================================================================================
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
WHERE m.room_type = 'All' AND m.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city);


-- ====================================================================================================
-- ds_room_mix   (Room type mix and median price per room type.)
-- ====================================================================================================
SELECT
  initcap(m.city)                      AS city,
  m.room_type,
  m.listings,
  round(100 * m.share_of_listings, 1)  AS pct_of_city_listings,
  round(m.median_price_eur, 1)         AS median_price_eur,
  CASE WHEN m.city IN ('berlin', 'vienna') THEN '1' ELSE '0' END AS is_ab
FROM workspace.airbnb.gold_city_metrics m
WHERE m.room_type <> 'All' AND m.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city);


-- ====================================================================================================
-- ds_price_bins   (Price distribution: share of priced listings per 25 EUR band (500+ in one band).)
-- ====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
b AS (
  SELECT f.city, least(floor(f.price_eur / 25) * 25, 500) AS band_start
  FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  WHERE f.is_trainable
)
SELECT initcap(city) AS city,
       CASE WHEN band_start >= 500 THEN '500+' ELSE concat(lpad(cast(cast(band_start AS INT) AS STRING), 3, '0'), '-', cast(cast(band_start + 25 AS INT) AS STRING)) END AS price_band_eur,
       count(*) AS listings,
       round(100 * count(*) / sum(count(*)) OVER (PARTITION BY city), 2) AS pct_of_city
FROM b GROUP BY city, band_start;


-- ====================================================================================================
-- ds_availability_bins   (Availability distribution: share of listings per 30-day band of availability_365.)
-- ====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
b AS (
  SELECT f.city, least(floor(f.availability_365 / 30) * 30, 360) AS band_start
  FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  WHERE f.availability_365 IS NOT NULL
)
SELECT initcap(city) AS city,
       concat(lpad(cast(cast(band_start AS INT) AS STRING), 3, '0'), '-', cast(cast(least(band_start + 29, 365) AS INT) AS STRING)) AS days_available,
       count(*) AS listings,
       round(100 * count(*) / sum(count(*)) OVER (PARTITION BY city), 2) AS pct_of_city
FROM b GROUP BY city, band_start;


-- ====================================================================================================
-- ds_neighbourhoods_top   (Top 10 neighbourhoods per city by median price and by number of listings (20+ priced listings).)
-- ====================================================================================================
WITH n AS (
  SELECT n.*,
         row_number() OVER (PARTITION BY n.city ORDER BY n.median_price_eur DESC) AS price_rank,
         row_number() OVER (PARTITION BY n.city ORDER BY n.listings DESC)         AS size_rank
  FROM workspace.airbnb.gold_neighbourhood_metrics n
  WHERE n.listings_with_price >= 20
    AND n.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_neighbourhood_metrics x WHERE x.city = n.city)
)
SELECT initcap(city) AS city, neighbourhood, listings, round(median_price_eur, 1) AS median_price_eur,
       price_rank, size_rank,
       CASE WHEN price_rank <= 10 THEN '1' ELSE '0' END AS price_rank_ok,
       CASE WHEN size_rank <= 10 THEN '1' ELSE '0' END AS size_rank_ok,
       concat(initcap(city), ': ', neighbourhood) AS city_neighbourhood
FROM n WHERE price_rank <= 10 OR size_rank <= 10;


-- ====================================================================================================
-- ds_compare_ab   (Berlin against Vienna, one row per metric.)
-- ====================================================================================================
WITH c AS (
  SELECT m.city, CAST(m.listings AS DOUBLE) AS listings,
         m.median_price_eur, m.median_availability_365, m.avg_review_rating,
         m.listings / m.hosts AS listings_per_host, m.share_long_stay, m.share_superhost,
         1 - m.listings_with_price / m.listings AS share_without_price
  FROM workspace.airbnb.gold_city_metrics m
  WHERE m.room_type = 'All' AND m.city IN ('berlin', 'vienna') AND m.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city)
),
p AS (
  SELECT
    max(CASE WHEN city = 'berlin' THEN listings END) AS b_listings, max(CASE WHEN city = 'vienna' THEN listings END) AS v_listings,
    max(CASE WHEN city = 'berlin' THEN median_price_eur END) AS b_price, max(CASE WHEN city = 'vienna' THEN median_price_eur END) AS v_price,
    max(CASE WHEN city = 'berlin' THEN median_availability_365 END) AS b_avail, max(CASE WHEN city = 'vienna' THEN median_availability_365 END) AS v_avail,
    max(CASE WHEN city = 'berlin' THEN avg_review_rating END) AS b_rating, max(CASE WHEN city = 'vienna' THEN avg_review_rating END) AS v_rating,
    max(CASE WHEN city = 'berlin' THEN listings_per_host END) AS b_lph, max(CASE WHEN city = 'vienna' THEN listings_per_host END) AS v_lph,
    max(CASE WHEN city = 'berlin' THEN share_long_stay END) AS b_long, max(CASE WHEN city = 'vienna' THEN share_long_stay END) AS v_long,
    max(CASE WHEN city = 'berlin' THEN share_superhost END) AS b_super, max(CASE WHEN city = 'vienna' THEN share_superhost END) AS v_super,
    max(CASE WHEN city = 'berlin' THEN share_without_price END) AS b_noprice, max(CASE WHEN city = 'vienna' THEN share_without_price END) AS v_noprice
  FROM c
)
SELECT metric, round(berlin, 2) AS berlin, round(vienna, 2) AS vienna, round(vienna - berlin, 2) AS vienna_minus_berlin
FROM p
LATERAL VIEW stack(8,
  '1 Listings', b_listings, v_listings,
  '2 Median price (EUR)', b_price, v_price,
  '3 Median availability (days/year)', b_avail, v_avail,
  '4 Avg review score (0-5)', b_rating, v_rating,
  '5 Listings per host', b_lph, v_lph,
  '6 Long-stay listings (%)', 100 * b_long, 100 * v_long,
  '7 Superhosts (%)', 100 * b_super, 100 * v_super,
  '8 Listings without price (%)', 100 * b_noprice, 100 * v_noprice
) s AS metric, berlin, vienna;


-- ====================================================================================================
-- ds_model_metrics   (MAE of model and baseline from the model notebooks (all segments).)
-- ====================================================================================================
SELECT
  model_version,
  CASE WHEN city = 'berlin+vienna' THEN 'Berlin + Vienna' ELSE initcap(city) END AS city,
  CASE WHEN evaluated_on = 'test' THEN 'test set (20 % of listings)'
       WHEN model_version = 'v1' THEN 'all priced listings (first run)'
       ELSE 'held-out listings only' END AS evaluated_on,
  segment, n_listings, mae_baseline_eur, mae_model_eur, improvement_pct, median_ae_eur, bias_eur, median_price_eur
FROM workspace.airbnb.gold_model_metrics;


-- ====================================================================================================
-- ds_model_mae_long   (Headline MAE per city, one row per predictor (baseline or model), for a grouped bar.)
-- ====================================================================================================
SELECT model_version,
       CASE WHEN city = 'berlin+vienna' THEN 'Berlin + Vienna' ELSE initcap(city) END AS city,
       predictor, mae_eur
FROM workspace.airbnb.gold_model_metrics
LATERAL VIEW stack(2, 'baseline', mae_baseline_eur, 'model', mae_model_eur) s AS predictor, mae_eur
WHERE segment = 'all';


-- ====================================================================================================
-- ds_predictions   (One row per evaluated listing and model version (see EVAL_ROWS).)
-- ====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
)
SELECT
  p.model_version,
  initcap(f.city)                                     AS city,
  CASE WHEN f.city IN ('berlin', 'vienna') THEN 'test set (20 % of listings)'
              WHEN p.model_version = 'v1' THEN 'all priced listings (first run)'
              ELSE 'held-out listings only' END AS evaluated_on,
  f.listing_id,
  f.neighbourhood,
  f.room_type,
  f.accommodates,
  CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
  round(f.dist_to_centre_km, 2)                       AS dist_to_centre_km,
  f.price_eur                                         AS actual_price_eur,
  p.predicted_price_eur,
  round(p.predicted_price_eur - f.price_eur, 2)       AS error_eur,
  round(abs(p.predicted_price_eur - f.price_eur), 2)  AS abs_error_eur,
  CASE WHEN p.predicted_price_eur < f.price_eur THEN 'under-priced' ELSE 'over-priced' END AS direction,
  concat('https://www.airbnb.com/rooms/', f.listing_id) AS listing_url
FROM workspace.airbnb.gold_listing_features f
JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'));


-- ====================================================================================================
-- ds_error_segments   (MAE and bias by room type and by stay type, per model version and city.)
-- ====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
e AS (
  SELECT p.model_version, f.city, f.room_type,
         CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
         p.predicted_price_eur - f.price_eur AS err
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
)
SELECT model_version, initcap(city) AS city, 'room type' AS dimension, room_type AS segment,
       count(*) AS n_listings, round(avg(abs(err)), 2) AS mae_eur, round(avg(err), 2) AS bias_eur
FROM e GROUP BY 1, 2, 3, 4
UNION ALL
SELECT model_version, initcap(city), 'stay type', stay_type, count(*), round(avg(abs(err)), 2), round(avg(err), 2)
FROM e GROUP BY 1, 2, 3, 4;


-- ====================================================================================================
-- ds_error_by_neighbourhood   (Error per neighbourhood (10+ evaluated listings), with rank of most under-priced per city.)
-- ====================================================================================================
SELECT *, CASE WHEN under_rank <= 10 THEN '1' ELSE '0' END AS top10 FROM (
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
e AS (
  SELECT p.model_version, f.city, f.neighbourhood, f.price_eur, p.predicted_price_eur
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
),
g AS (
  SELECT model_version, city, neighbourhood, count(*) AS n_listings,
         round(percentile(price_eur, 0.5), 1) AS median_actual_eur,
         round(percentile(predicted_price_eur, 0.5), 1) AS median_predicted_eur,
         round(avg(abs(predicted_price_eur - price_eur)), 2) AS mae_eur,
         round(avg(predicted_price_eur - price_eur), 2) AS bias_eur
  FROM e GROUP BY model_version, city, neighbourhood
  HAVING count(*) >= 10
)
SELECT model_version, initcap(city) AS city, neighbourhood,
       concat(initcap(city), ': ', neighbourhood) AS city_neighbourhood,
       n_listings, median_actual_eur, median_predicted_eur, mae_eur, bias_eur,
       row_number() OVER (PARTITION BY model_version, city ORDER BY bias_eur ASC) AS under_rank
FROM g
) t;


-- ====================================================================================================
-- ds_extreme_errors   (The 20 most under-priced and 20 most over-priced listings per model version and city, in EUR.)
-- ====================================================================================================
WITH p AS (
  WITH latest AS (
    SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
  )
  SELECT
    p.model_version,
    initcap(f.city)                                     AS city,
    CASE WHEN f.city IN ('berlin', 'vienna') THEN 'test set (20 % of listings)'
                WHEN p.model_version = 'v1' THEN 'all priced listings (first run)'
                ELSE 'held-out listings only' END AS evaluated_on,
    f.listing_id,
    f.neighbourhood,
    f.room_type,
    f.accommodates,
    CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
    round(f.dist_to_centre_km, 2)                       AS dist_to_centre_km,
    f.price_eur                                         AS actual_price_eur,
    p.predicted_price_eur,
    round(p.predicted_price_eur - f.price_eur, 2)       AS error_eur,
    round(abs(p.predicted_price_eur - f.price_eur), 2)  AS abs_error_eur,
    CASE WHEN p.predicted_price_eur < f.price_eur THEN 'under-priced' ELSE 'over-priced' END AS direction,
    concat('https://www.airbnb.com/rooms/', f.listing_id) AS listing_url
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
),
r AS (
  SELECT p.*,
         row_number() OVER (PARTITION BY model_version, city, direction
                            ORDER BY CASE WHEN direction = 'under-priced' THEN error_eur ELSE -error_eur END) AS rnk
  FROM p
)
SELECT model_version, city, direction, rnk, neighbourhood, room_type, accommodates, stay_type,
       actual_price_eur, predicted_price_eur, error_eur, listing_url
FROM r WHERE rnk <= 20;


-- ====================================================================================================
-- ds_munich_like_for_like   (Munich, the same held-out listings (is_test): baseline, v1 and v2 side by side.)
-- ====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
m AS (   -- Munich held-out listings: the only rows on which v1, v2 and the baseline can be compared fairly
  SELECT f.listing_id, f.neighbourhood, f.price_eur
  FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  WHERE f.city = 'munich' AND f.is_trainable AND f.is_test
),
e AS (
  SELECT 'baseline' AS predictor, m.listing_id, m.neighbourhood, m.price_eur, b.baseline_price_eur AS pred
  FROM m JOIN workspace.airbnb.gold_baseline_predictions b ON b.city = 'munich' AND b.listing_id = m.listing_id
  UNION ALL
  SELECT concat('model ', p.model_version), m.listing_id, m.neighbourhood, m.price_eur, p.predicted_price_eur
  FROM m JOIN workspace.airbnb.gold_price_predictions p ON p.city = 'munich' AND p.listing_id = m.listing_id
)
SELECT predictor, count(*) AS n_listings,
       round(avg(abs(pred - price_eur)), 2) AS mae_eur,
       round(avg(pred - price_eur), 2)      AS bias_eur
FROM e GROUP BY predictor;


-- ====================================================================================================
-- ds_munich_neighbourhoods   (Munich by neighbourhood on the held-out listings: error of baseline, v1 and v2 (10+ listings).)
-- ====================================================================================================
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
m AS (   -- Munich held-out listings: the only rows on which v1, v2 and the baseline can be compared fairly
  SELECT f.listing_id, f.neighbourhood, f.price_eur
  FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  WHERE f.city = 'munich' AND f.is_trainable AND f.is_test
),
e AS (
  SELECT 'baseline' AS predictor, m.listing_id, m.neighbourhood, m.price_eur, b.baseline_price_eur AS pred
  FROM m JOIN workspace.airbnb.gold_baseline_predictions b ON b.city = 'munich' AND b.listing_id = m.listing_id
  UNION ALL
  SELECT concat('model ', p.model_version), m.listing_id, m.neighbourhood, m.price_eur, p.predicted_price_eur
  FROM m JOIN workspace.airbnb.gold_price_predictions p ON p.city = 'munich' AND p.listing_id = m.listing_id
)
SELECT neighbourhood,
       count(DISTINCT listing_id)                                                        AS n_listings,
       round(percentile(price_eur, 0.5), 1)                                              AS median_price_eur,
       round(avg(CASE WHEN predictor = 'model v1' THEN abs(pred - price_eur) END), 1)    AS mae_v1_eur,
       round(avg(CASE WHEN predictor = 'model v2' THEN abs(pred - price_eur) END), 1)    AS mae_v2_eur,
       round(avg(CASE WHEN predictor = 'baseline' THEN abs(pred - price_eur) END), 1)    AS mae_baseline_eur,
       round(avg(CASE WHEN predictor = 'model v1' THEN pred - price_eur END), 1)         AS bias_v1_eur,
       round(avg(CASE WHEN predictor = 'model v2' THEN pred - price_eur END), 1)         AS bias_v2_eur
FROM e GROUP BY neighbourhood
HAVING count(DISTINCT listing_id) >= 10;


-- ====================================================================================================
-- ds_neighbourhoods_top__price_rank_ok_1   (Top 10 neighbourhoods per city by median price and by number of listings (20+ priced listings). Only rows where price_rank_ok = '1'.)
-- ====================================================================================================
SELECT * FROM (
WITH n AS (
  SELECT n.*,
         row_number() OVER (PARTITION BY n.city ORDER BY n.median_price_eur DESC) AS price_rank,
         row_number() OVER (PARTITION BY n.city ORDER BY n.listings DESC)         AS size_rank
  FROM workspace.airbnb.gold_neighbourhood_metrics n
  WHERE n.listings_with_price >= 20
    AND n.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_neighbourhood_metrics x WHERE x.city = n.city)
)
SELECT initcap(city) AS city, neighbourhood, listings, round(median_price_eur, 1) AS median_price_eur,
       price_rank, size_rank,
       CASE WHEN price_rank <= 10 THEN '1' ELSE '0' END AS price_rank_ok,
       CASE WHEN size_rank <= 10 THEN '1' ELSE '0' END AS size_rank_ok,
       concat(initcap(city), ': ', neighbourhood) AS city_neighbourhood
FROM n WHERE price_rank <= 10 OR size_rank <= 10
) t
WHERE price_rank_ok = '1';


-- ====================================================================================================
-- ds_neighbourhoods_top__size_rank_ok_1   (Top 10 neighbourhoods per city by median price and by number of listings (20+ priced listings). Only rows where size_rank_ok = '1'.)
-- ====================================================================================================
SELECT * FROM (
WITH n AS (
  SELECT n.*,
         row_number() OVER (PARTITION BY n.city ORDER BY n.median_price_eur DESC) AS price_rank,
         row_number() OVER (PARTITION BY n.city ORDER BY n.listings DESC)         AS size_rank
  FROM workspace.airbnb.gold_neighbourhood_metrics n
  WHERE n.listings_with_price >= 20
    AND n.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_neighbourhood_metrics x WHERE x.city = n.city)
)
SELECT initcap(city) AS city, neighbourhood, listings, round(median_price_eur, 1) AS median_price_eur,
       price_rank, size_rank,
       CASE WHEN price_rank <= 10 THEN '1' ELSE '0' END AS price_rank_ok,
       CASE WHEN size_rank <= 10 THEN '1' ELSE '0' END AS size_rank_ok,
       concat(initcap(city), ': ', neighbourhood) AS city_neighbourhood
FROM n WHERE price_rank <= 10 OR size_rank <= 10
) t
WHERE size_rank_ok = '1';


-- ====================================================================================================
-- ds_room_mix__is_ab_1   (Room type mix and median price per room type. Only rows where is_ab = '1'.)
-- ====================================================================================================
SELECT * FROM (
SELECT
  initcap(m.city)                      AS city,
  m.room_type,
  m.listings,
  round(100 * m.share_of_listings, 1)  AS pct_of_city_listings,
  round(m.median_price_eur, 1)         AS median_price_eur,
  CASE WHEN m.city IN ('berlin', 'vienna') THEN '1' ELSE '0' END AS is_ab
FROM workspace.airbnb.gold_city_metrics m
WHERE m.room_type <> 'All' AND m.snapshot_date = (SELECT max(snapshot_date) FROM workspace.airbnb.gold_city_metrics x WHERE x.city = m.city)
) t
WHERE is_ab = '1';


-- ====================================================================================================
-- ds_ab_segments__dimension_stay_type   (Berlin and Vienna: median price by stay type and by distance band (priced listings). Only rows where dimension = 'stay type'.)
-- ====================================================================================================
SELECT * FROM (
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
)
SELECT initcap(f.city) AS city, 'stay type' AS dimension,
       CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS segment,
       count(*) AS listings, round(percentile(f.price_eur, 0.5), 1) AS median_price_eur
FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
WHERE f.is_trainable AND f.city IN ('berlin', 'vienna')
GROUP BY 1, 2, 3
UNION ALL
SELECT initcap(f.city), 'distance to centre',
       CASE WHEN f.dist_to_centre_km < 2 THEN 'a) 0-2 km' WHEN f.dist_to_centre_km < 5 THEN 'b) 2-5 km'
            WHEN f.dist_to_centre_km < 10 THEN 'c) 5-10 km' ELSE 'd) 10+ km' END,
       count(*), round(percentile(f.price_eur, 0.5), 1)
FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
WHERE f.is_trainable AND f.city IN ('berlin', 'vienna')
GROUP BY 1, 2, 3
) t
WHERE dimension = 'stay type';


-- ====================================================================================================
-- ds_ab_segments__dimension_distance_to_centre   (Berlin and Vienna: median price by stay type and by distance band (priced listings). Only rows where dimension = 'distance to centre'.)
-- ====================================================================================================
SELECT * FROM (
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
)
SELECT initcap(f.city) AS city, 'stay type' AS dimension,
       CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS segment,
       count(*) AS listings, round(percentile(f.price_eur, 0.5), 1) AS median_price_eur
FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
WHERE f.is_trainable AND f.city IN ('berlin', 'vienna')
GROUP BY 1, 2, 3
UNION ALL
SELECT initcap(f.city), 'distance to centre',
       CASE WHEN f.dist_to_centre_km < 2 THEN 'a) 0-2 km' WHEN f.dist_to_centre_km < 5 THEN 'b) 2-5 km'
            WHEN f.dist_to_centre_km < 10 THEN 'c) 5-10 km' ELSE 'd) 10+ km' END,
       count(*), round(percentile(f.price_eur, 0.5), 1)
FROM workspace.airbnb.gold_listing_features f JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
WHERE f.is_trainable AND f.city IN ('berlin', 'vienna')
GROUP BY 1, 2, 3
) t
WHERE dimension = 'distance to centre';


-- ====================================================================================================
-- ds_model_metrics__segment_all   (MAE of model and baseline from the model notebooks (all segments). Only rows where segment = 'all'.)
-- ====================================================================================================
SELECT * FROM (
SELECT
  model_version,
  CASE WHEN city = 'berlin+vienna' THEN 'Berlin + Vienna' ELSE initcap(city) END AS city,
  CASE WHEN evaluated_on = 'test' THEN 'test set (20 % of listings)'
       WHEN model_version = 'v1' THEN 'all priced listings (first run)'
       ELSE 'held-out listings only' END AS evaluated_on,
  segment, n_listings, mae_baseline_eur, mae_model_eur, improvement_pct, median_ae_eur, bias_eur, median_price_eur
FROM workspace.airbnb.gold_model_metrics
) t
WHERE segment = 'all';


-- ====================================================================================================
-- ds_error_segments__dimension_room_type   (MAE and bias by room type and by stay type, per model version and city. Only rows where dimension = 'room type'.)
-- ====================================================================================================
SELECT * FROM (
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
e AS (
  SELECT p.model_version, f.city, f.room_type,
         CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
         p.predicted_price_eur - f.price_eur AS err
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
)
SELECT model_version, initcap(city) AS city, 'room type' AS dimension, room_type AS segment,
       count(*) AS n_listings, round(avg(abs(err)), 2) AS mae_eur, round(avg(err), 2) AS bias_eur
FROM e GROUP BY 1, 2, 3, 4
UNION ALL
SELECT model_version, initcap(city), 'stay type', stay_type, count(*), round(avg(abs(err)), 2), round(avg(err), 2)
FROM e GROUP BY 1, 2, 3, 4
) t
WHERE dimension = 'room type';


-- ====================================================================================================
-- ds_error_segments__dimension_stay_type   (MAE and bias by room type and by stay type, per model version and city. Only rows where dimension = 'stay type'.)
-- ====================================================================================================
SELECT * FROM (
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
e AS (
  SELECT p.model_version, f.city, f.room_type,
         CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
         p.predicted_price_eur - f.price_eur AS err
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
)
SELECT model_version, initcap(city) AS city, 'room type' AS dimension, room_type AS segment,
       count(*) AS n_listings, round(avg(abs(err)), 2) AS mae_eur, round(avg(err), 2) AS bias_eur
FROM e GROUP BY 1, 2, 3, 4
UNION ALL
SELECT model_version, initcap(city), 'stay type', stay_type, count(*), round(avg(abs(err)), 2), round(avg(err), 2)
FROM e GROUP BY 1, 2, 3, 4
) t
WHERE dimension = 'stay type';


-- ====================================================================================================
-- ds_error_by_neighbourhood__top10_1   (Error per neighbourhood (10+ evaluated listings), with rank of most under-priced per city. Only rows where top10 = '1'.)
-- ====================================================================================================
SELECT * FROM (
SELECT *, CASE WHEN under_rank <= 10 THEN '1' ELSE '0' END AS top10 FROM (
WITH latest AS (
  SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
),
e AS (
  SELECT p.model_version, f.city, f.neighbourhood, f.price_eur, p.predicted_price_eur
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
),
g AS (
  SELECT model_version, city, neighbourhood, count(*) AS n_listings,
         round(percentile(price_eur, 0.5), 1) AS median_actual_eur,
         round(percentile(predicted_price_eur, 0.5), 1) AS median_predicted_eur,
         round(avg(abs(predicted_price_eur - price_eur)), 2) AS mae_eur,
         round(avg(predicted_price_eur - price_eur), 2) AS bias_eur
  FROM e GROUP BY model_version, city, neighbourhood
  HAVING count(*) >= 10
)
SELECT model_version, initcap(city) AS city, neighbourhood,
       concat(initcap(city), ': ', neighbourhood) AS city_neighbourhood,
       n_listings, median_actual_eur, median_predicted_eur, mae_eur, bias_eur,
       row_number() OVER (PARTITION BY model_version, city ORDER BY bias_eur ASC) AS under_rank
FROM g
) t
) t
WHERE top10 = '1';


-- ====================================================================================================
-- ds_extreme_errors__direction_under_priced   (The 20 most under-priced and 20 most over-priced listings per model version and city, in EUR. Only rows where direction = 'under-priced'.)
-- ====================================================================================================
SELECT * FROM (
WITH p AS (
  WITH latest AS (
    SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
  )
  SELECT
    p.model_version,
    initcap(f.city)                                     AS city,
    CASE WHEN f.city IN ('berlin', 'vienna') THEN 'test set (20 % of listings)'
                WHEN p.model_version = 'v1' THEN 'all priced listings (first run)'
                ELSE 'held-out listings only' END AS evaluated_on,
    f.listing_id,
    f.neighbourhood,
    f.room_type,
    f.accommodates,
    CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
    round(f.dist_to_centre_km, 2)                       AS dist_to_centre_km,
    f.price_eur                                         AS actual_price_eur,
    p.predicted_price_eur,
    round(p.predicted_price_eur - f.price_eur, 2)       AS error_eur,
    round(abs(p.predicted_price_eur - f.price_eur), 2)  AS abs_error_eur,
    CASE WHEN p.predicted_price_eur < f.price_eur THEN 'under-priced' ELSE 'over-priced' END AS direction,
    concat('https://www.airbnb.com/rooms/', f.listing_id) AS listing_url
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
),
r AS (
  SELECT p.*,
         row_number() OVER (PARTITION BY model_version, city, direction
                            ORDER BY CASE WHEN direction = 'under-priced' THEN error_eur ELSE -error_eur END) AS rnk
  FROM p
)
SELECT model_version, city, direction, rnk, neighbourhood, room_type, accommodates, stay_type,
       actual_price_eur, predicted_price_eur, error_eur, listing_url
FROM r WHERE rnk <= 20
) t
WHERE direction = 'under-priced';


-- ====================================================================================================
-- ds_extreme_errors__direction_over_priced   (The 20 most under-priced and 20 most over-priced listings per model version and city, in EUR. Only rows where direction = 'over-priced'.)
-- ====================================================================================================
SELECT * FROM (
WITH p AS (
  WITH latest AS (
    SELECT city, max(snapshot_date) AS snapshot_date FROM workspace.airbnb.gold_listing_features GROUP BY city
  )
  SELECT
    p.model_version,
    initcap(f.city)                                     AS city,
    CASE WHEN f.city IN ('berlin', 'vienna') THEN 'test set (20 % of listings)'
                WHEN p.model_version = 'v1' THEN 'all priced listings (first run)'
                ELSE 'held-out listings only' END AS evaluated_on,
    f.listing_id,
    f.neighbourhood,
    f.room_type,
    f.accommodates,
    CASE WHEN f.is_long_stay THEN 'long stay (30+ nights)' ELSE 'short stay' END AS stay_type,
    round(f.dist_to_centre_km, 2)                       AS dist_to_centre_km,
    f.price_eur                                         AS actual_price_eur,
    p.predicted_price_eur,
    round(p.predicted_price_eur - f.price_eur, 2)       AS error_eur,
    round(abs(p.predicted_price_eur - f.price_eur), 2)  AS abs_error_eur,
    CASE WHEN p.predicted_price_eur < f.price_eur THEN 'under-priced' ELSE 'over-priced' END AS direction,
    concat('https://www.airbnb.com/rooms/', f.listing_id) AS listing_url
  FROM workspace.airbnb.gold_listing_features f
  JOIN latest l ON f.city = l.city AND f.snapshot_date = l.snapshot_date
  JOIN workspace.airbnb.gold_price_predictions p ON p.city = f.city AND p.listing_id = f.listing_id
  WHERE f.is_trainable AND (f.is_test OR (f.city NOT IN ('berlin', 'vienna') AND p.model_version = 'v1'))
),
r AS (
  SELECT p.*,
         row_number() OVER (PARTITION BY model_version, city, direction
                            ORDER BY CASE WHEN direction = 'under-priced' THEN error_eur ELSE -error_eur END) AS rnk
  FROM p
)
SELECT model_version, city, direction, rnk, neighbourhood, room_type, accommodates, stay_type,
       actual_price_eur, predicted_price_eur, error_eur, listing_url
FROM r WHERE rnk <= 20
) t
WHERE direction = 'over-priced';
