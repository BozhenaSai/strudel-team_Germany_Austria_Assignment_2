# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Checks — fail the Job if the output cannot be trusted
# MAGIC
# MAGIC Runs last. Every check prints what it found; any failure raises, so the Job turns red instead of publishing a wrong table quietly.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("city", "vienna")
dbutils.widgets.text("snapshot_date", "2026-09-17")
dbutils.widgets.text("currency", "EUR")

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

def run_checks():
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(("PASS  " if ok else "FAIL  ") + name + (f"  - {detail}" if detail else ""))

    silver = in_scope(spark.table(SILVER))
    features = in_scope(spark.table(GOLD_FEATURES))
    n_silver, n_features = silver.count(), features.count()

    check("silver has rows for this city", n_silver > 0, f"{n_silver:,} rows")
    check("silver: one row per listing_id",
          silver.select("listing_id").distinct().count() == n_silver)
    check("gold features: one row per silver listing", n_features == n_silver, f"{n_features:,} vs {n_silver:,}")
    check("gold features: no NULL keys",
          features.filter("listing_id IS NULL OR city IS NULL").count() == 0)

    leaked = [c for c in LEAKAGE_COLUMNS if c in spark.table(GOLD_FEATURES).columns]
    check("gold features: no leakage columns", not leaked, ", ".join(leaked))

    trainable = features.filter("is_trainable").count()
    check("gold features: some trainable listings", trainable > 0, f"{trainable:,} trainable")
    check("is_trainable never includes a listing without a price",
          features.filter("is_trainable AND price_eur IS NULL").count() == 0)

    log = in_scope(spark.table(RUN_LOG)).filter("stage = 'silver' AND status = 'ok'").orderBy(F.desc("run_ts")).limit(1).collect()
    if log:
        r = log[0]
        check("run log: received = silver + quarantined + duplicates",
              r["rows_in"] == r["rows_out"] + r["rows_quarantined"] + r["duplicates_removed"],
              f"{r['rows_in']:,} = {r['rows_out']:,} + {r['rows_quarantined']:,} + {r['duplicates_removed']:,}")
    else:
        check("run log has a silver entry for this run", False)

    failed = [n for n, ok, _ in results if not ok]
    log_run("checks", rows_in=n_silver, rows_out=n_features, status="ok" if not failed else "FAILED",
            note="; ".join(failed) if failed else f"{len(results)} checks passed")
    if failed:
        raise AssertionError("Failed checks: " + "; ".join(failed))


run_checks()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run log (latest first)

# COMMAND ----------

display(spark.table(RUN_LOG).orderBy(F.desc("run_ts")).limit(40))