# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 · Silver — typed, cleaned, one row per listing
# MAGIC
# MAGIC Applies the cleaning rules from the EDA (`eda/eda_berlin_vienna.ipynb`, section 5). Grain is preserved: **one row = one listing on the snapshot date.**
# MAGIC
# MAGIC | EDA step | Where it happens |
# MAGIC |---|---|
# MAGIC | 1 add city + snapshot date | `city`, `snapshot_date`, `currency` |
# MAGIC | 2 price text -> number | `price_eur` |
# MAGIC | 3 keep listings without a price, mark them | `price_missing`, `outlier_class = 'no_price'` |
# MAGIC | 4 outliers per city x room type, on log price | `outlier_class` |
# MAGIC | 5 `t`/`f` -> boolean | `host_is_superhost`, `has_availability` |
# MAGIC | 6 bathrooms from `bathrooms_text`, shared flag | `bathrooms`, `bathrooms_shared` |
# MAGIC | 7 amenities -> count + flags | `n_amenities`, `has_*` |
# MAGIC | 8 broken characters in district names | `neighbourhood` |
# MAGIC | 9 date text -> dates | `last_scraped`, `first_review`, `last_review` |
# MAGIC | 10 select by name, never fail on a missing column | `num()`, `s()`, ... helpers |
# MAGIC | 12 long-stay flag | `is_long_stay` |
# MAGIC
# MAGIC Steps 11 and 13 (remove leakage, impute) belong to model features and are done in `03_gold`.
# MAGIC
# MAGIC **Quarantine, not drop** (Lecture 4): a row that is structurally unusable (no id, impossible coordinates, no room type, no guests) goes to
# MAGIC `quarantine_listings` with `_failed_rules` attached. Listings without a price and price outliers are **not** broken rows, they stay in silver with flags.
# MAGIC
# MAGIC Silver deliberately does not carry free text, URLs, or the price-derived columns (`estimated_*`, `price_quote_*`); they stay in bronze.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Column helpers: select by name, tolerate a missing column
# MAGIC
# MAGIC Serverless compute runs with ANSI mode on, where a plain `cast` of a bad string *raises*. Everything below uses `try_cast`, which returns NULL instead, so one malformed value becomes a NULL that the rules can see, not a crashed job.

# COMMAND ----------

def make_helpers(df):
    cols = set(df.columns)

    def s(name):
        """Trimmed string, empty -> NULL."""
        if name not in cols:
            return F.lit(None).cast("string")
        t = F.trim(F.col(name))
        return F.when(F.length(t) > 0, t)

    def num(name, dtype="double", exact=False):
        """Number or NULL. Integer columns go via double so '2.0' still parses; ids use exact=True (a double would lose digits)."""
        if name not in cols:
            return F.lit(None).cast(dtype)
        if dtype in ("int", "bigint") and not exact:
            return F.expr(f"try_cast(try_cast(`{name}` as double) as {dtype})")
        return F.expr(f"try_cast(`{name}` as {dtype})")

    def money(name):
        """'$1,234.00' -> 1234.0   (the sign is a dollar sign in every file, the value is in the region's currency)"""
        if name not in cols:
            return F.lit(None).cast("double")
        return F.expr(f"try_cast(regexp_replace(`{name}`, '[^0-9.-]', '') as double)")

    def tf(name):
        """'t'/'f' -> boolean, anything else -> NULL."""
        if name not in cols:
            return F.lit(None).cast("boolean")
        return F.when(F.col(name) == "t", F.lit(True)).when(F.col(name) == "f", F.lit(False))

    def dt(name):
        if name not in cols:
            return F.lit(None).cast("date")
        return F.expr(f"try_cast(`{name}` as date)")

    return s, num, money, tf, dt


def fix_mojibake(x):
    """Some districts arrive decoded with the wrong code page (Mac Roman read as Latin-1), e.g. 'W\x8ahring'.

    Only strings that contain the tell-tale characters are touched, so correct names are left alone.
    """
    if isinstance(x, str) and any(("\x80" <= ch <= "\x9f") or ch == "§" for ch in x):
        try:
            return x.encode("latin-1").decode("mac_roman")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return x
    return x


fix_mojibake_udf = F.udf(fix_mojibake, "string")

# amenity name patterns, matched on the lower-cased JSON text; the leading quote keeps "dishwasher" out of "washer"
AMENITY_FLAGS = {
    "has_kitchen": r'"kitchen(ette)?"',
    "has_washer": r'"(free |paid )?washer',
    "has_air_conditioning": r"air conditioning",
    "has_wifi": r'"(fast )?wifi',
    "has_parking": r"parking",
    "has_elevator": r'"elevator"',
    "has_balcony": r"patio or balcony",
    "has_dedicated_workspace": r"dedicated workspace",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Typing

# COMMAND ----------

def build_typed(raw):
    s, num, money, tf, dt = make_helpers(raw)

    # bathrooms: numeric column if the export has it again, else parse "1.5 shared baths" / "Half-bath"
    text = F.lower(s("bathrooms_text"))
    digits = F.regexp_extract(text, r"(\d+(?:\.\d+)?)", 1)
    parsed_bath = F.coalesce(
        F.when(F.length(digits) > 0, digits).cast("double"),
        F.when(text.contains("half"), F.lit(0.5)),
    )

    # amenities: JSON list stored as text
    amen = s("amenities")
    amen_list = F.from_json(amen, "array<string>")
    amen_lower = F.lower(amen)

    cols = [
        F.col("_city").alias("city"),
        F.col("_snapshot_date").alias("snapshot_date"),
        F.lit(CURRENCY).alias("currency"),
        num("id", "bigint", exact=True).alias("listing_id"),
        num("host_id", "bigint", exact=True).alias("host_id"),
        fix_mojibake_udf(s("neighbourhood_cleansed")).alias("neighbourhood"),
        s("neighbourhood_group_cleansed").alias("neighbourhood_group"),
        num("latitude").alias("latitude"),
        num("longitude").alias("longitude"),
        s("property_type").alias("property_type"),
        s("room_type").alias("room_type"),
        num("accommodates", "int").alias("accommodates"),
        num("bedrooms").alias("bedrooms"),
        num("beds").alias("beds"),
        F.coalesce(num("bathrooms"), parsed_bath).alias("bathrooms"),
        text.contains("shared").alias("bathrooms_shared"),
        F.when(amen_list.isNotNull(), F.size(amen_list)).alias("n_amenities"),
        *[amen_lower.rlike(p).alias(name) for name, p in AMENITY_FLAGS.items()],
        money("price").alias("price_eur"),
        num("minimum_nights", "int").alias("minimum_nights"),
        num("maximum_nights", "bigint").alias("maximum_nights"),
        tf("has_availability").alias("has_availability"),
        num("availability_30", "int").alias("availability_30"),
        num("availability_60", "int").alias("availability_60"),
        num("availability_90", "int").alias("availability_90"),
        num("availability_365", "int").alias("availability_365"),
        num("number_of_reviews", "int").alias("number_of_reviews"),
        num("number_of_reviews_ltm", "int").alias("number_of_reviews_ltm"),
        num("reviews_per_month").alias("reviews_per_month"),
        num("review_scores_rating").alias("review_scores_rating"),
        num("review_scores_accuracy").alias("review_scores_accuracy"),
        num("review_scores_cleanliness").alias("review_scores_cleanliness"),
        num("review_scores_checkin").alias("review_scores_checkin"),
        num("review_scores_communication").alias("review_scores_communication"),
        num("review_scores_location").alias("review_scores_location"),
        num("review_scores_value").alias("review_scores_value"),
        dt("first_review").alias("first_review"),
        dt("last_review").alias("last_review"),
        tf("host_is_superhost").alias("host_is_superhost"),
        num("calculated_host_listings_count", "int").alias("calculated_host_listings_count"),
        dt("last_scraped").alias("last_scraped"),
        F.col("_ingested_at"),
        F.col("_source"),
    ]
    return raw.select(*cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rules and quarantine
# MAGIC
# MAGIC Same `split_valid` as Lab 4, including the `coalesce`: a rule that evaluates to NULL counts as a failure, otherwise the row would land in neither table and vanish silently.
# MAGIC
# MAGIC The rules state only what must be true of *any* listing in *any* city. They are deliberately not fitted to Berlin or Vienna.

# COMMAND ----------

SILVER_RULES = {
    "has_id":              "listing_id IS NOT NULL",
    "valid_coordinates":   "latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180",
    "has_room_type":       "room_type IS NOT NULL",
    "valid_accommodates":  "accommodates >= 1",
}


def split_valid(df, rules: dict):
    conds = [F.coalesce(F.expr(c), F.lit(False)).alias(n) for n, c in rules.items()]
    tagged = df.select("*", *conds)

    all_pass = F.lit(True)
    for name in rules:
        all_pass = all_pass & F.col(name)
    failed = F.concat_ws(",", *[F.when(~F.col(n), F.lit(n)) for n in rules])

    good = tagged.filter(all_pass).drop(*rules.keys())
    bad = (tagged.filter(~all_pass)
           .withColumn("_failed_rules", failed)
           .withColumn("_quarantined_at", F.current_timestamp())
           .drop(*rules.keys()))
    return good, bad

# COMMAND ----------

# MAGIC %md
# MAGIC ## De-duplication and flags

# COMMAND ----------

def dedupe(good):
    """One row per listing: if an id appears twice keep the most recently scraped one."""
    w = (Window.partitionBy("city", "snapshot_date", "listing_id")
         .orderBy(F.col("last_scraped").desc_nulls_last(), F.col("_ingested_at").desc()))
    return good.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def add_flags(df):
    """Flags the later stages rely on. Outliers are judged per city x room type on log price, so the rule has no fixed cut-off
    and works at a different price level in a new city."""
    priced = df.filter(F.col("price_eur") > 0).withColumn("_log_price", F.log("price_eur"))
    stats = priced.groupBy("city", "room_type").agg(
        F.expr("percentile(_log_price, 0.25)").alias("_q1"),
        F.expr("percentile(_log_price, 0.75)").alias("_q3"),
    )
    j = (df.join(stats, ["city", "room_type"], "left")
           .withColumn("_log_price", F.when(F.col("price_eur") > 0, F.log("price_eur")))
           .withColumn("_iqr", F.col("_q3") - F.col("_q1")))

    outlier_class = (
        F.when(F.col("price_eur").isNull(), "no_price")
        .when(F.col("price_eur") < MIN_PLAUSIBLE_PRICE, "implausible_low")
        .when(F.col("_log_price") > F.col("_q3") + OUTLIER_IQR_FACTOR * F.col("_iqr"), "extreme_high")
        .when(F.col("_log_price") < F.col("_q1") - OUTLIER_IQR_FACTOR * F.col("_iqr"), "extreme_low")
        .otherwise("ok")
    )
    return (j
            .withColumn("price_missing", F.col("price_eur").isNull())
            .withColumn("outlier_class", outlier_class)
            .withColumn("is_long_stay", F.coalesce(F.col("minimum_nights") >= LONG_STAY_NIGHTS, F.lit(False)))
            .withColumn("has_reviews", F.coalesce(F.col("number_of_reviews") > 0, F.lit(False)))
            .drop("_log_price", "_q1", "_q3", "_iqr"))

# COMMAND ----------

def run_silver():
    raw = in_scope(spark.table(BRONZE), "_city", "_snapshot_date")
    n_in = raw.count()
    if n_in == 0:
        raise ValueError(f"No bronze rows for {CITY} / {SNAPSHOT_DATE}. Run 01_bronze first.")

    typed = build_typed(raw)
    good, bad = split_valid(typed, SILVER_RULES)
    n_good = good.count()

    silver = add_flags(dedupe(good))
    write_city_partition(silver, SILVER)
    write_city_partition(bad, QUARANTINE)

    n_silver = in_scope(spark.table(SILVER)).count()
    n_bad = in_scope(spark.table(QUARANTINE)).count()
    n_dup = n_good - n_silver

    print(f"rows received   {n_in:>8,}")
    print(f"silver          {n_silver:>8,}")
    print(f"quarantined     {n_bad:>8,}")
    print(f"duplicates      {n_dup:>8,}")
    balanced = n_in == n_silver + n_bad + n_dup
    print(f"books balance:  {balanced}   (received = silver + quarantined + duplicates)")
    if not balanced:
        raise AssertionError("Rows were lost between bronze and silver")

    log_run("silver", rows_in=n_in, rows_out=n_silver, rows_quarantined=n_bad, duplicates_removed=n_dup)


with logged_stage("silver"):
    run_silver()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What silver set aside, and how it flagged the rest

# COMMAND ----------

display(in_scope(spark.table(QUARANTINE)).groupBy("_failed_rules").count())

# COMMAND ----------

display(
    in_scope(spark.table(SILVER))
    .groupBy("outlier_class")
    .agg(F.count("*").alias("listings"), F.round(F.avg("price_eur"), 1).alias("avg_price_eur"))
    .orderBy(F.desc("listings"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Missing values by column, so a new city's gaps are visible on the first run:

# COMMAND ----------

_s = in_scope(spark.table(SILVER))
display(
    _s.select([F.round(F.avg(F.col(c).isNull().cast("int")), 3).alias(c) for c in _s.columns if not c.startswith("_")])
      .toPandas().T.rename(columns={0: "share_missing"}).sort_values("share_missing", ascending=False).reset_index()
      .rename(columns={"index": "column"})
)