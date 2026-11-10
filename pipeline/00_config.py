# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Config — shared by every stage of the pipeline
# MAGIC
# MAGIC Not run on its own: `01_bronze`, `02_silver`, `03_gold` and `04_checks` pull it in with `%run ./00_config`.
# MAGIC
# MAGIC **One pipeline, the city is a parameter.** Nothing in the stages is hard-coded to Berlin or Vienna.
# MAGIC Add a city by dropping `<city>/listings.csv.gz` into the Volume and running the Job with new parameters.
# MAGIC
# MAGIC | Parameter | Example | Meaning |
# MAGIC |---|---|---|
# MAGIC | `city` | `berlin` | folder name under `/Volumes/workspace/airbnb/raw/` |
# MAGIC | `snapshot_date` | `2026-09-19` | Inside Airbnb snapshot date (the folder date on the website) |
# MAGIC | `currency` | `EUR` | currency of the `price` column (all cities of a region share it) |

# COMMAND ----------

import re
from contextlib import contextmanager
from datetime import date, datetime, timezone

from pyspark.sql import functions as F

# ---- where things live ------------------------------------------------------------------
CATALOG = "workspace"
SCHEMA = "airbnb"
RAW_BASE = f"/Volumes/{CATALOG}/{SCHEMA}/raw"      # <RAW_BASE>/<city>/listings.csv.gz


def fq(table: str) -> str:
    """Fully qualified table name."""
    return f"{CATALOG}.{SCHEMA}.{table}"


BRONZE = fq("bronze_listings")
SILVER = fq("silver_listings")
QUARANTINE = fq("quarantine_listings")
GOLD_FEATURES = fq("gold_listing_features")
GOLD_CITY_METRICS = fq("gold_city_metrics")
GOLD_NEIGHBOURHOOD_METRICS = fq("gold_neighbourhood_metrics")
GOLD_CITY_CENTRE = fq("gold_city_centre")
RUN_LOG = fq("pipeline_run_log")

# ---- job parameters (a Job overrides these defaults) --------------------------------------
dbutils.widgets.text("city", "berlin", "City (folder under raw/)")
dbutils.widgets.text("snapshot_date", "2026-09-19", "Snapshot date YYYY-MM-DD")
dbutils.widgets.text("currency", "EUR", "Currency of the price column")

CITY = dbutils.widgets.get("city").strip().lower()
SNAPSHOT_DATE = dbutils.widgets.get("snapshot_date").strip()
CURRENCY = dbutils.widgets.get("currency").strip().upper()

if not re.fullmatch(r"[a-z][a-z0-9_]*", CITY):
    raise ValueError(f"city must be a lowercase folder name, got {CITY!r}")
date.fromisoformat(SNAPSHOT_DATE)          # raises ValueError if it is not YYYY-MM-DD

# ---- rules that come from the EDA (eda/eda_berlin_vienna.ipynb) ----------------------------
MIN_PLAUSIBLE_PRICE = 15.0     # below this a nightly price is a placeholder / wrong unit (EUR; rescale for other currencies)
OUTLIER_IQR_FACTOR = 3         # extreme price = beyond Q1/Q3 -/+ 3 IQR of log price, per city x room type
LONG_STAY_NIGHTS = 30          # minimum_nights >= 30 -> long-stay listing (priced very differently)
TEST_SHARE_PCT = 20            # ~20% of listings per city are flagged is_test (deterministic hash of the id)

# Columns that reveal the price (or whether a price exists) and must never be model features.
# 04_checks fails the Job if any of them shows up in the gold feature table.
LEAKAGE_COLUMNS = [
    "estimated_revenue_l365d", "estimated_occupancy_l365d",
    "price_quote_price_per_night", "price_quote_total_price", "price_quote_raw",
    "price_quote_checkin_date", "price_quote_checkout_date",
    "availability_30", "availability_60", "availability_90",
]

print(f"city={CITY}  snapshot_date={SNAPSHOT_DATE}  currency={CURRENCY}")
print(f"raw file: {RAW_BASE}/{CITY}/listings.csv.gz")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------

def table_exists(table: str) -> bool:
    return spark.catalog.tableExists(table)


def write_city_partition(df, table: str, city_col: str = "city", date_col: str = "snapshot_date"):
    """Idempotent write: replace only this city+snapshot, leave every other city untouched.

    Re-running the pipeline for Berlin never duplicates Berlin and never touches Vienna.
    """
    writer = df.write.format("delta").option("mergeSchema", "true")
    if table_exists(table):
        writer = (writer.mode("overwrite")
                  .option("replaceWhere", f"{city_col} = '{CITY}' AND {date_col} = DATE'{SNAPSHOT_DATE}'"))
    else:
        writer = writer.mode("overwrite")          # first ever run: creates the table
    writer.saveAsTable(table)


def in_scope(df, city_col: str = "city", date_col: str = "snapshot_date"):
    """Rows of the city + snapshot this run is about."""
    return df.filter((F.col(city_col) == CITY) & (F.col(date_col) == F.lit(SNAPSHOT_DATE).cast("date")))


_LOG_SCHEMA = ("run_ts timestamp, city string, snapshot_date date, stage string, status string, "
               "rows_in long, rows_out long, rows_quarantined long, duplicates_removed long, note string")


def log_run(stage, rows_in=0, rows_out=0, rows_quarantined=0, duplicates_removed=0, status="ok", note=""):
    """Append one line to pipeline_run_log: what came in, what came out, what was set aside."""
    row = [(datetime.now(timezone.utc).replace(tzinfo=None), CITY, date.fromisoformat(SNAPSHOT_DATE), stage, status,
            int(rows_in), int(rows_out), int(rows_quarantined), int(duplicates_removed), note)]
    (spark.createDataFrame(row, _LOG_SCHEMA)
          .write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(RUN_LOG))


@contextmanager
def logged_stage(stage):
    """Wrap a stage so a crash is written to pipeline_run_log before it propagates.

    That is how a first run on a new city leaves a written record of what broke.
    """
    try:
        yield
    except Exception as e:                                   # noqa: BLE001 - re-raised below
        log_run(stage, status="FAILED", note=f"{type(e).__name__}: {str(e)[:900]}")
        raise
