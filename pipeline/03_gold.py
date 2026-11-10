# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Gold — model features, city metrics, descriptions for Genie
# MAGIC
# MAGIC Everything here is built from `silver_listings` for the city + snapshot being run, and **replaces only that city's rows**.
# MAGIC
# MAGIC | Table | Grain | Used by |
# MAGIC |---|---|---|
# MAGIC | `gold_listing_features` | one row per listing | the model (participant 3), baseline, dashboard, Genie |
# MAGIC | `gold_city_metrics` | city x snapshot x room type (+ an `All` row) | dashboard pages 1-2, Genie |
# MAGIC | `gold_neighbourhood_metrics` | city x snapshot x neighbourhood | dashboard, Genie |
# MAGIC | `gold_city_centre` | city x snapshot | explains `dist_to_centre_km` |
# MAGIC
# MAGIC **Leakage is removed here** (EDA step 11): no `estimated_*`, no `price_quote_*`, no `availability_30/60/90`, no ids, no free text. `04_checks` fails the Job if one slips in.
# MAGIC
# MAGIC **Location without district names.** Names differ in every city, so the feature is `dist_to_centre_km`: haversine distance to the *median coordinate of the city's listings*. It needs no per-city configuration, so a new city works unchanged.
# MAGIC
# MAGIC **`is_test`** = ~20% of listings per city, picked by a hash of the id, identical on every run. The model and the baseline must use the same flag. For a new client city (C) every listing is unseen: evaluate on all rows, ignore `is_test`.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("city", "vienna")
dbutils.widgets.text("snapshot_date", "2026-09-17")
dbutils.widgets.text("currency", "EUR")

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

def haversine_km(lat, lon, lat0, lon0):
    """Great-circle distance in km from (lat, lon) columns to a fixed point."""
    r = 6371.0088
    dlat = F.radians(lat - F.lit(lat0))
    dlon = F.radians(lon - F.lit(lon0))
    a = (F.sin(dlat / 2) ** 2
         + F.cos(F.radians(F.lit(lat0))) * F.cos(F.radians(lat)) * F.sin(dlon / 2) ** 2)
    return F.lit(2 * r) * F.asin(F.sqrt(F.least(F.lit(1.0), a)))


FEATURE_COMMENTS = {
    "city": "City the listing belongs to (lower case).",
    "snapshot_date": "Inside Airbnb snapshot date of the file the listing came from.",
    "listing_id": "Airbnb listing id. One row per listing. Key only, never a model feature.",
    "neighbourhood": "District name (broken characters fixed). Descriptive only: names differ per city, do not use as a model feature.",
    "room_type": "Entire home/apt, Private room, Hotel room or Shared room.",
    "property_type": "Airbnb property type, free-form label.",
    "accommodates": "Maximum number of guests.",
    "bedrooms": "Number of bedrooms; missing values filled with the city median for the same number of guests.",
    "beds": "Number of beds; missing values filled with the city median for the same number of guests.",
    "bathrooms": "Number of bathrooms parsed from the bathroom description (half-bath = 0.5).",
    "bathrooms_shared": "True if the bathroom is shared.",
    "minimum_nights": "Minimum stay in nights.",
    "is_long_stay": "True if minimum stay is 30 nights or more. Long-stay listings cost about half as much per night.",
    "n_amenities": "Number of amenities listed.",
    "has_kitchen": "Amenity flag: kitchen or kitchenette.",
    "has_washer": "Amenity flag: washer (free, paid, in unit or in building).",
    "has_air_conditioning": "Amenity flag: air conditioning.",
    "has_wifi": "Amenity flag: wifi.",
    "has_parking": "Amenity flag: any parking.",
    "has_elevator": "Amenity flag: elevator.",
    "has_balcony": "Amenity flag: patio or balcony.",
    "has_dedicated_workspace": "Amenity flag: dedicated workspace.",
    "has_reviews": "True if the listing has at least one review.",
    "number_of_reviews": "Total number of reviews.",
    "number_of_reviews_ltm": "Reviews in the last twelve months.",
    "reviews_per_month": "Average reviews per month; 0 for listings without reviews.",
    "review_scores_rating": "Overall review score 0-5; listings without reviews get the city median (see has_reviews).",
    "review_scores_location": "Location review score 0-5; listings without reviews get the city median (see has_reviews).",
    "host_is_superhost": "True if the host is a superhost.",
    "calculated_host_listings_count": "Number of listings of the same host.",
    "availability_365": "Days the listing is open in the next 365 days. Sign of how active it is.",
    "latitude": "Latitude in degrees.",
    "longitude": "Longitude in degrees.",
    "dist_to_centre_km": "Distance in km to the city centre, defined as the median coordinate of the city's listings (see gold_city_centre).",
    "price_eur": "TARGET. Nightly price advertised on the snapshot date, in the region's currency (EUR). NULL if the listing had no price.",
    "price_missing": "True if the listing had no price on the snapshot date (mostly listings that cannot be booked).",
    "outlier_class": "no_price, implausible_low (<15), extreme_high, extreme_low (3 IQR of log price within city and room type) or ok.",
    "is_trainable": "True if the listing has a price and outlier_class is ok. Train and evaluate on these rows only.",
    "is_test": "True for about 20% of listings per city (hash of the id, stable between runs). For a new client city ignore it: every row is unseen.",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features

# COMMAND ----------

def build_features(silver, lat0, lon0):
    by_guests = silver.groupBy("accommodates").agg(
        F.expr("percentile(beds, 0.5)").alias("_beds_med"),
        F.expr("percentile(bedrooms, 0.5)").alias("_bedrooms_med"),
    )
    city_med = silver.agg(
        F.expr("percentile(beds, 0.5)").alias("_beds_city"),
        F.expr("percentile(bedrooms, 0.5)").alias("_bedrooms_city"),
        F.expr("percentile(review_scores_rating, 0.5)").alias("_rating_city"),
        F.expr("percentile(review_scores_location, 0.5)").alias("_rating_loc_city"),
    )
    j = silver.join(by_guests, "accommodates", "left").crossJoin(city_med)

    return j.select(
        "city", "snapshot_date", "listing_id", "neighbourhood",
        # --- target and its flags ---
        "price_eur", "price_missing", "outlier_class",
        ((~F.col("price_missing")) & (F.col("outlier_class") == "ok")).alias("is_trainable"),
        (F.pmod(F.xxhash64("city", "listing_id"), F.lit(100)) < TEST_SHARE_PCT).alias("is_test"),
        # --- features (known for any listing, not derived from the price) ---
        "room_type", "property_type", "accommodates",
        F.coalesce("bedrooms", F.round("_bedrooms_med"), F.round("_bedrooms_city")).alias("bedrooms"),
        F.coalesce("beds", F.round("_beds_med"), F.round("_beds_city")).alias("beds"),
        "bathrooms", "bathrooms_shared",
        "minimum_nights", "is_long_stay",
        "n_amenities", "has_kitchen", "has_washer", "has_air_conditioning", "has_wifi",
        "has_parking", "has_elevator", "has_balcony", "has_dedicated_workspace",
        "has_reviews",
        F.coalesce("number_of_reviews", F.lit(0)).alias("number_of_reviews"),
        F.coalesce("number_of_reviews_ltm", F.lit(0)).alias("number_of_reviews_ltm"),
        F.coalesce("reviews_per_month", F.lit(0.0)).alias("reviews_per_month"),
        F.coalesce("review_scores_rating", "_rating_city").alias("review_scores_rating"),
        F.coalesce("review_scores_location", "_rating_loc_city").alias("review_scores_location"),
        F.coalesce("host_is_superhost", F.lit(False)).alias("host_is_superhost"),
        "calculated_host_listings_count", "availability_365",
        "latitude", "longitude",
        haversine_km(F.col("latitude"), F.col("longitude"), lat0, lon0).alias("dist_to_centre_km"),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics for the dashboard and Genie

# COMMAND ----------

def metric_aggs():
    return [
        F.count("*").alias("listings"),
        F.count("price_eur").alias("listings_with_price"),
        F.expr("percentile(price_eur, 0.5)").alias("median_price_eur"),
        F.avg("availability_365").alias("avg_availability_365"),
        F.expr("percentile(availability_365, 0.5)").alias("median_availability_365"),
        F.avg("review_scores_rating").alias("avg_review_rating"),
        F.countDistinct("host_id").alias("hosts"),
        F.expr("percentile(accommodates, 0.5)").alias("median_accommodates"),
        F.avg(F.col("is_long_stay").cast("int")).alias("share_long_stay"),
        F.avg(F.col("host_is_superhost").cast("int")).alias("share_superhost"),
    ]


def build_city_metrics(silver):
    keys = ["city", "snapshot_date"]
    total = silver.count()
    by_type = silver.groupBy(*keys, "room_type").agg(*metric_aggs())
    overall = silver.groupBy(*keys).agg(*metric_aggs()).withColumn("room_type", F.lit("All"))
    return (by_type.unionByName(overall)
            .withColumn("share_of_listings", F.col("listings") / F.lit(total))
            .select(*keys, "room_type", "listings", "share_of_listings", "listings_with_price", "median_price_eur",
                    "avg_availability_365", "median_availability_365", "avg_review_rating", "hosts",
                    "median_accommodates", "share_long_stay", "share_superhost"))


def build_neighbourhood_metrics(silver):
    return (silver.groupBy("city", "snapshot_date", "neighbourhood")
            .agg(F.count("*").alias("listings"),
                 F.count("price_eur").alias("listings_with_price"),
                 F.expr("percentile(price_eur, 0.5)").alias("median_price_eur"),
                 F.avg("review_scores_rating").alias("avg_review_rating"),
                 F.avg("availability_365").alias("avg_availability_365"),
                 F.avg((F.col("room_type") == "Entire home/apt").cast("int")).alias("share_entire_home")))

# COMMAND ----------

def apply_comments(table, comments):
    """Table + column descriptions: they are what Genie and the dashboard read to understand the data."""
    existing = {f.name for f in spark.table(table).schema.fields}
    for col, text in comments.items():
        if col in existing:
            safe = text.replace("'", "''")
            spark.sql(f"ALTER TABLE {table} ALTER COLUMN `{col}` COMMENT '{safe}'")


def run_gold():
    silver = in_scope(spark.table(SILVER))
    n_silver = silver.count()
    if n_silver == 0:
        raise ValueError(f"No silver rows for {CITY} / {SNAPSHOT_DATE}. Run 02_silver first.")

    # city centre = median coordinate of the city's listings (no per-city configuration needed)
    c = silver.agg(F.expr("percentile(latitude, 0.5)").alias("lat"),
                   F.expr("percentile(longitude, 0.5)").alias("lon")).first()
    lat0, lon0 = float(c["lat"]), float(c["lon"])
    centre = spark.createDataFrame(
        [(CITY, date.fromisoformat(SNAPSHOT_DATE), lat0, lon0, "median coordinate of the city's listings")],
        "city string, snapshot_date date, centre_lat double, centre_lon double, method string")
    write_city_partition(centre, GOLD_CITY_CENTRE)

    write_city_partition(build_features(silver, lat0, lon0), GOLD_FEATURES)
    write_city_partition(build_city_metrics(silver), GOLD_CITY_METRICS)
    write_city_partition(build_neighbourhood_metrics(silver), GOLD_NEIGHBOURHOOD_METRICS)

    spark.sql(f"COMMENT ON TABLE {GOLD_FEATURES} IS 'One row per listing: target price_eur, leakage-free model features, is_trainable and is_test flags. Built by the parameterised pipeline.'")
    spark.sql(f"COMMENT ON TABLE {GOLD_CITY_METRICS} IS 'Listing metrics per city and room type (room_type = All is the whole city). Median price, availability, review score, hosts, room type mix.'")
    spark.sql(f"COMMENT ON TABLE {GOLD_NEIGHBOURHOOD_METRICS} IS 'Listing metrics per city and neighbourhood: number of listings, median price, review score.'")
    spark.sql(f"COMMENT ON TABLE {GOLD_CITY_CENTRE} IS 'City centre used for dist_to_centre_km: median coordinate of the city listings.'")
    apply_comments(GOLD_FEATURES, FEATURE_COMMENTS)

    n_features = in_scope(spark.table(GOLD_FEATURES)).count()
    print(f"gold_listing_features: {n_features:,} rows (silver: {n_silver:,})  centre = ({lat0:.4f}, {lon0:.4f})")
    if n_features != n_silver:
        raise AssertionError("gold_listing_features must have one row per silver listing")
    log_run("gold", rows_in=n_silver, rows_out=n_features,
            note=f"centre=({lat0:.4f},{lon0:.4f})")


with logged_stage("gold"):
    run_gold()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick look

# COMMAND ----------

display(in_scope(spark.table(GOLD_CITY_METRICS)).orderBy("room_type"))

# COMMAND ----------

display(
    in_scope(spark.table(GOLD_FEATURES))
    .agg(F.count("*").alias("listings"),
         F.sum(F.col("is_trainable").cast("int")).alias("trainable"),
         F.round(100 * F.avg(F.col("is_test").cast("int")), 1).alias("pct_is_test"),
         F.round(F.expr("percentile(dist_to_centre_km, 0.5)"), 2).alias("median_dist_to_centre_km"))
)