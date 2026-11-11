# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Bronze — capture exactly what arrived
# MAGIC
# MAGIC Lecture 4 rule: *everything the source sent, plus metadata. Nothing parsed, nothing filtered, nothing fixed.*
# MAGIC
# MAGIC * every one of the ~90 source columns is kept **as text**, so a column that changes shape in a new city cannot break the load;
# MAGIC * four metadata columns say what arrived, from where and when: `_city`, `_snapshot_date`, `_ingested_at`, `_source`;
# MAGIC * `multiLine` is required: `host_about` contains line breaks inside quoted fields;
# MAGIC * a re-run **replaces only this city + snapshot**, so it never duplicates rows and never touches other cities.
# MAGIC
# MAGIC Job parameters: `city`, `snapshot_date`.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

def run_bronze():
    src = f"{RAW_BASE}/{CITY}/listings.csv.gz"
    try:
        dbutils.fs.ls(src)
    except Exception as e:
        raise FileNotFoundError(
            f"No listings file for city '{CITY}'. Expected {src} - upload it to the Volume first.") from e

    raw = (
        spark.read.format("csv")
        .option("header", True)
        .option("multiLine", True)            # line breaks inside quoted text fields
        .option("quote", '"')
        .option("escape", '"')                # Inside Airbnb escapes a quote by doubling it
        .option("inferSchema", False)         # bronze keeps everything as text
        .load(src)
    )

    bronze = (
        raw
        .withColumn("_city", F.lit(CITY))
        .withColumn("_snapshot_date", F.lit(SNAPSHOT_DATE).cast("date"))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source", F.lit(src))
    )
    write_city_partition(bronze, BRONZE, city_col="_city", date_col="_snapshot_date")

    n = in_scope(spark.table(BRONZE), "_city", "_snapshot_date").count()
    print(f"bronze_listings: {n:,} rows for {CITY} / {SNAPSHOT_DATE}  ({len(raw.columns)} source columns)")
    if n == 0:
        raise ValueError(f"{src} was read but produced 0 rows")
    log_run("bronze", rows_in=n, rows_out=n, note=f"{len(raw.columns)} source columns, kept as text")


with logged_stage("bronze"):
    run_bronze()

# COMMAND ----------

# MAGIC %md
# MAGIC What bronze can now answer about itself: what arrived, from where, when.

# COMMAND ----------

display(
    spark.table(BRONZE)
    .groupBy("_city", "_snapshot_date", "_source")
    .agg(F.count("*").alias("rows_received"), F.max("_ingested_at").alias("last_ingested"))
    .orderBy("_city")
)