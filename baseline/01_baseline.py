# Databricks notebook source
# MAGIC %md
# MAGIC # Baseline: median price by room type and guests
# MAGIC
# MAGIC The baseline the model must beat (assignment 3.4):
# MAGIC the median `price_eur` for the same `room_type` and `accommodates`, computed on the **train rows only**
# MAGIC (`is_trainable AND NOT is_test`, Berlin and Vienna together).
# MAGIC
# MAGIC A combination that is not in train falls back to the median of the room type, then to the overall median.
# MAGIC This matters most for Munich, where some (room type, guests) pairs never appear in Berlin or Vienna.
# MAGIC
# MAGIC This notebook is written in Spark, separately from `model/00_model_common.py` (pandas).
# MAGIC Its purpose is to check that the two give the same numbers, so the model is compared against one agreed baseline.
# MAGIC
# MAGIC Output: `gold_baseline_predictions` (city, listing_id, baseline_price_eur, baseline_level) for every listing,
# MAGIC so the dashboard can show where the baseline and the model disagree.

# COMMAND ----------

dbutils.widgets.text("train_cities", "berlin,vienna", "Training cities")
dbutils.widgets.text("new_cities", "munich", "New-client cities (evaluated on all trainable rows)")

from pyspark.sql import functions as F, Window

FEATURES = "workspace.airbnb.gold_listing_features"
METRICS = "workspace.airbnb.gold_model_metrics"
OUT = "workspace.airbnb.gold_baseline_predictions"

TRAIN_CITIES = [c.strip().lower() for c in dbutils.widgets.get("train_cities").split(",") if c.strip()]
NEW_CITIES = [c.strip().lower() for c in dbutils.widgets.get("new_cities").split(",") if c.strip()]

# latest snapshot of every city
latest = F.max("snapshot_date").over(Window.partitionBy("city"))
features = (spark.table(FEATURES)
            .withColumn("_latest", latest)
            .filter(F.col("snapshot_date") == F.col("_latest")).drop("_latest"))

train = features.filter(F.col("city").isin(TRAIN_CITIES) & F.col("is_trainable") & ~F.col("is_test"))
print(f"train rows: {train.count():,}  cities: {TRAIN_CITIES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit and predict

# COMMAND ----------

def median(col):
    # exact median, the same as pandas .median()
    return F.expr(f"percentile({col}, 0.5)")


def fit_baseline(train_df):
    """Three lookup tables: (room_type, accommodates) -> median, room_type -> median, overall median."""
    return {
        "by_room_guests": train_df.groupBy("room_type", "accommodates").agg(median("price_eur").alias("m_room_guests")),
        "by_room": train_df.groupBy("room_type").agg(median("price_eur").alias("m_room")),
        "overall": train_df.agg(median("price_eur").alias("m_overall")),
    }


def predict_baseline(df, b):
    """Adds baseline_price_eur and baseline_level (which fallback was used) to any listing table."""
    out = (df.join(b["by_room_guests"], ["room_type", "accommodates"], "left")
             .join(b["by_room"], ["room_type"], "left")
             .crossJoin(b["overall"]))
    return (out
            .withColumn("baseline_price_eur", F.coalesce("m_room_guests", "m_room", "m_overall"))
            .withColumn("baseline_level",
                        F.when(F.col("m_room_guests").isNotNull(), "room_type + accommodates")
                         .when(F.col("m_room").isNotNull(), "room_type only")
                         .otherwise("overall"))
            .drop("m_room_guests", "m_room", "m_overall"))


def mae_table(df, group_col):
    """MAE in EUR per group plus a total row. df needs price_eur and baseline_price_eur."""
    err = F.abs(F.col("baseline_price_eur") - F.col("price_eur"))
    per_group = df.groupBy(group_col).agg(F.count("*").alias("n"), F.round(F.avg(err), 2).alias("mae_baseline_eur"))
    groups = sorted(r[0] for r in df.select(group_col).distinct().collect())
    if len(groups) < 2:
        return per_group
    total = df.agg(F.count("*").alias("n"), F.round(F.avg(err), 2).alias("mae_baseline_eur")) \
              .withColumn(group_col, F.lit("+".join(groups)))   # "berlin+vienna", the label the model notebook uses
    return per_group.unionByName(total)


baseline = fit_baseline(train)
scored = predict_baseline(features, baseline)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MAE on the test rows of the training cities

# COMMAND ----------

test = scored.filter(F.col("city").isin(TRAIN_CITIES) & F.col("is_trainable") & F.col("is_test"))
test_mae = mae_table(test, "city")
display(test_mae)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MAE on the new-client cities (every trainable row, `is_test` ignored)
# MAGIC `baseline_level` shows how many listings needed a fallback because their (room type, guests) pair is not in train.

# COMMAND ----------

new = scored.filter(F.col("city").isin(NEW_CITIES) & F.col("is_trainable"))
if new.limit(1).count() > 0:
    display(mae_table(new, "city"))
    display(new.groupBy("city", "baseline_level").count().orderBy("city", "baseline_level"))
else:
    print(f"No rows yet for {NEW_CITIES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check: same numbers as the model notebook
# MAGIC `gold_model_metrics` holds the baseline MAE computed by `model/00_model_common.py`.
# MAGIC Every value must match to the cent, or the two definitions differ.

# COMMAND ----------

mine = test_mae.unionByName(mae_table(new, "city") if new.limit(1).count() > 0 else test_mae.limit(0))
theirs = (spark.table(METRICS)
          .filter((F.col("model_version") == "v1") & (F.col("segment") == "all"))
          .select("city", "evaluated_on", F.col("mae_baseline_eur").alias("mae_baseline_model_nb")))
check = mine.join(theirs, "city", "left").withColumn(
    "match", F.abs(F.col("mae_baseline_eur") - F.col("mae_baseline_model_nb")) < 0.01)
display(check.orderBy("city"))

bad = check.filter(F.col("mae_baseline_model_nb").isNotNull() & ~F.col("match")).count()
assert bad == 0, "Baseline MAE differs from gold_model_metrics: the two baseline definitions are not the same"
print("Baseline matches gold_model_metrics.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write baseline predictions for every listing
# MAGIC Re-running replaces the table. One row per (city, listing_id), also for listings without a price.

# COMMAND ----------

(scored.select("city", "listing_id",
               F.round("baseline_price_eur", 2).alias("baseline_price_eur"),
               "baseline_level")
 .write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT))

spark.sql(f"COMMENT ON TABLE {OUT} IS 'Baseline price per listing: median price_eur of the train rows (Berlin and Vienna, "
          f"is_trainable and not is_test) for the same room_type and accommodates, falling back to the room type median, "
          f"then the overall median. Join to gold_listing_features on (city, listing_id).'")
for col, text in {
    "baseline_price_eur": "Baseline predicted price per night in EUR. The model must beat this.",
    "baseline_level": "Which median was used: room_type + accommodates, room_type only (pair not in train) or overall.",
}.items():
    spark.sql(f"ALTER TABLE {OUT} ALTER COLUMN `{col}` COMMENT '{text}'")

print(f"{OUT}: {spark.table(OUT).count():,} rows")
