# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Model common — shared by training and scoring
# MAGIC
# MAGIC Not run on its own: `01_train_evaluate` and `02_score_city` pull it in with `%run ./00_model_common`.
# MAGIC
# MAGIC Everything that must be **identical** at training time and at scoring time lives here: the feature lists,
# MAGIC the preprocessing, the baseline and the metric. `02_score_city` never fits anything, it only applies
# MAGIC what `01_train_evaluate` saved. That is what makes the city C run "unchanged".
# MAGIC
# MAGIC Input: `workspace.airbnb.gold_listing_features`. Rules we follow from the pipeline:
# MAGIC
# MAGIC * target `price_eur`; train and evaluate only where `is_trainable = true`;
# MAGIC * train = `is_test = false`, test = `is_test = true` (the shared split, we never make our own);
# MAGIC * for a new city `is_test` is ignored: every trainable row is unseen;
# MAGIC * never a feature: `listing_id`, `snapshot_date`, `neighbourhood`, `price_missing`, `outlier_class`, `is_trainable`, `is_test`.

# COMMAND ----------

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---- where things live --------------------------------------------------------------------
CATALOG = "workspace"
SCHEMA = "airbnb"


def fq(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"


GOLD_FEATURES = fq("gold_listing_features")
GOLD_PREDICTIONS = fq("gold_price_predictions")
GOLD_MODEL_METRICS = fq("gold_model_metrics")
MODEL_VOLUME = "models"                                   # Unity Catalog volume for the saved model
MODEL_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{MODEL_VOLUME}"

TARGET = "price_eur"
SEED = 42

# ---- features -----------------------------------------------------------------------------
# All of them come from the allowed list of the pipeline. None is derived from the price.
CAT_BASE = ["room_type", "property_type"]
NUM_BASE = [
    "accommodates", "bedrooms", "beds", "bathrooms",                      # size: the main price driver (EDA 4c)
    "minimum_nights",                                                     # long stays are cheaper per night
    "n_amenities",                                                        # how well equipped the place is
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",    # demand / how established it is
    "review_scores_rating", "review_scores_location",                     # quality and location as guests rate it
    "calculated_host_listings_count",                                     # private host or professional operator
    "availability_365",                                                   # how active the listing is
    "dist_to_centre_km",                                                  # location, defined the same way in every city
]
BOOL_BASE = [
    "bathrooms_shared", "is_long_stay", "has_kitchen", "has_washer", "has_air_conditioning", "has_wifi",
    "has_parking", "has_elevator", "has_balcony", "has_dedicated_workspace", "has_reviews", "host_is_superhost",
]

# Feature sets, ordered from the most portable to the most city-specific.
#   core              nothing in it is tied to a particular city
#   core_no_avail     core without availability_365 (the EDA leakage review asked to test with and without it)
#   core_city         + city: lets the model learn a separate price level per city; a new city is an unseen value
#   core_city_latlon  + raw coordinates: can find several expensive areas inside a city (Berlin),
#                       but coordinates of a new city fall outside everything the model has seen
FEATURE_SETS = {
    "core_no_avail":    {"cat": CAT_BASE, "num": [c for c in NUM_BASE if c != "availability_365"] + BOOL_BASE},
    "core":             {"cat": CAT_BASE, "num": NUM_BASE + BOOL_BASE},
    "core_city":        {"cat": CAT_BASE + ["city"], "num": NUM_BASE + BOOL_BASE},
    "core_city_latlon": {"cat": CAT_BASE + ["city"], "num": NUM_BASE + BOOL_BASE + ["latitude", "longitude"]},
}
ALL_NUM = NUM_BASE + ["latitude", "longitude"]
ALL_CAT = CAT_BASE + ["city"]

MIN_PROPERTY_TYPE_COUNT = 30      # property types with fewer train listings are grouped into "Other"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing
# MAGIC Plain functions, no fitted state except the list of property types to keep, which is learned on the train rows
# MAGIC and stored with the model. A property type the model has never seen (likely in a new city) becomes `Other`.

# COMMAND ----------

def learn_property_types(train: pd.DataFrame) -> list:
    """Property types frequent enough in train to get their own category."""
    counts = train["property_type"].value_counts()
    return sorted(counts[counts >= MIN_PROPERTY_TYPE_COUNT].index.tolist())


def prepare(pdf: pd.DataFrame, property_types_kept: list) -> pd.DataFrame:
    """Same typing and grouping for train, test and any new city. Returns a copy."""
    out = pdf.copy()
    for c in ALL_NUM + [TARGET]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in BOOL_BASE:                                   # True/False/None -> 1.0/0.0/NaN
        out[c] = out[c].map({True: 1.0, False: 0.0, "true": 1.0, "false": 0.0}).astype(float)
    for c in ["is_trainable", "is_test"]:
        out[c] = out[c].fillna(False).astype(bool)
    out["property_type_raw"] = out["property_type"]
    out["property_type"] = out["property_type"].where(out["property_type"].isin(property_types_kept), "Other")
    for c in ALL_CAT:
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Models
# MAGIC Both predict **log price**; predictions are converted back with `exp`.
# MAGIC
# MAGIC * **Gradient boosting** (`HistGradientBoostingRegressor`) with absolute-error loss. The median does not change under
# MAGIC   `log`/`exp`, so the model predicts the median price for a listing, which is exactly what MAE rewards, and no
# MAGIC   correction is needed when converting back. Missing values are handled natively.
# MAGIC * **Ridge regression**: the simple, linear reference. If boosting did not beat it clearly we would keep Ridge.

# COMMAND ----------

GBM_PARAMS = dict(loss="absolute_error", learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
                  min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def make_gbm(feature_set: str) -> Pipeline:
    fs = FEATURE_SETS[feature_set]
    pre = ColumnTransformer([
        # unseen category (new property type, new city) -> NaN -> treated as "missing" by the trees
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), fs["cat"]),
        ("num", "passthrough", fs["num"]),
    ])
    model = HistGradientBoostingRegressor(categorical_features=list(range(len(fs["cat"]))), **GBM_PARAMS)
    return Pipeline([("pre", pre), ("model", model)])


def make_ridge(feature_set: str) -> Pipeline:
    fs = FEATURE_SETS[feature_set]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), fs["cat"]),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), fs["num"]),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])


# Candidates in order of preference when their cross-validated error is practically the same:
# simpler and more portable first.
CANDIDATES = [
    ("ridge | core",            make_ridge, "core"),
    ("gbm | core_no_avail",     make_gbm,   "core_no_avail"),
    ("gbm | core",              make_gbm,   "core"),
    ("gbm | core_city",         make_gbm,   "core_city"),
    ("gbm | core_city_latlon",  make_gbm,   "core_city_latlon"),
]


def feature_columns(feature_set: str) -> list:
    fs = FEATURE_SETS[feature_set]
    return fs["cat"] + fs["num"]


def fit_model(maker, feature_set: str, train: pd.DataFrame) -> Pipeline:
    pipe = maker(feature_set)
    pipe.fit(train[feature_columns(feature_set)], np.log(train[TARGET]))
    return pipe


def predict_price(pipe: Pipeline, feature_set: str, pdf: pd.DataFrame) -> np.ndarray:
    return np.exp(pipe.predict(pdf[feature_columns(feature_set)]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline
# MAGIC Median `price_eur` for the same `room_type` and `accommodates`, computed on the **train rows only**.
# MAGIC A combination that does not exist in train falls back to the median of the room type, then to the overall median.
# MAGIC
# MAGIC > Participant 4 owns the baseline. This is the same definition, kept here so the model notebook runs on its own.
# MAGIC > If participant 4's function differs, replace `fit_baseline` / `predict_baseline` and re-run: nothing else changes.

# COMMAND ----------

def fit_baseline(train: pd.DataFrame) -> dict:
    return {
        "by_room_guests": train.groupby(["room_type", "accommodates"])[TARGET].median().to_dict(),
        "by_room": train.groupby("room_type")[TARGET].median().to_dict(),
        "overall": float(train[TARGET].median()),
    }


def predict_baseline(baseline: dict, pdf: pd.DataFrame) -> np.ndarray:
    keys = list(zip(pdf["room_type"], pdf["accommodates"]))
    pred = pd.Series([baseline["by_room_guests"].get(k, np.nan) for k in keys], index=pdf.index, dtype=float)
    pred = pred.fillna(pdf["room_type"].map(baseline["by_room"]).astype(float))
    return pred.fillna(baseline["overall"]).to_numpy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metric
# MAGIC MAE in EUR (the required metric), plus two numbers that help explain an error:
# MAGIC `bias_eur` = mean of (predicted − actual): negative means the model under-prices on average;
# MAGIC `median_ae_eur` = the error of a typical listing, not pulled up by a few expensive ones.
# MAGIC
# MAGIC A small negative bias on the training cities is expected: the model predicts the *median* price of similar
# MAGIC listings, and the median of a right-skewed price is below its mean. A large negative bias in a new city is a
# MAGIC different thing: it means the whole city is more expensive than the model expects.

# COMMAND ----------

def error_row(actual, model_pred, baseline_pred) -> dict:
    actual, model_pred, baseline_pred = (np.asarray(a, dtype=float) for a in (actual, model_pred, baseline_pred))
    mae_m = float(np.mean(np.abs(model_pred - actual)))
    mae_b = float(np.mean(np.abs(baseline_pred - actual)))
    return {
        "n_listings": int(len(actual)),
        "mae_baseline_eur": round(mae_b, 2),
        "mae_model_eur": round(mae_m, 2),
        "improvement_pct": round(100 * (1 - mae_m / mae_b), 1) if mae_b > 0 else None,
        "median_ae_eur": round(float(np.median(np.abs(model_pred - actual))), 2),
        "bias_eur": round(float(np.mean(model_pred - actual)), 2),
        "median_price_eur": round(float(np.median(actual)), 2),
    }


def error_table(pdf: pd.DataFrame, by: str = "city", total_label: str = None) -> pd.DataFrame:
    """Errors per group of `by` (+ one total row). Needs columns price_eur, pred_model, pred_baseline."""
    rows = [{by: str(k), **error_row(g[TARGET], g["pred_model"], g["pred_baseline"])}
            for k, g in pdf.groupby(by, dropna=False)]
    if total_label:
        rows.append({by: total_label, **error_row(pdf[TARGET], pdf["pred_model"], pdf["pred_baseline"])})
    return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading and writing tables, saving and loading the model

# COMMAND ----------

def parse_cities(text: str) -> list:
    cities = [c.strip().lower() for c in text.split(",") if c.strip()]
    if not cities:
        raise ValueError("give at least one city")
    return cities


def load_features(cities: list) -> pd.DataFrame:
    """gold_listing_features for the given cities (latest snapshot of each) as pandas. ~25k rows: fits easily."""
    from pyspark.sql import functions as F, Window
    df = spark.table(GOLD_FEATURES).filter(F.col("city").isin(cities))
    latest = F.max("snapshot_date").over(Window.partitionBy("city"))
    pdf = df.withColumn("_latest", latest).filter(F.col("snapshot_date") == F.col("_latest")).drop("_latest").toPandas()
    missing = sorted(set(cities) - set(pdf["city"].unique()))
    if missing:
        raise ValueError(f"No rows in {GOLD_FEATURES} for {missing}. Has the pipeline run for these cities?")
    if pdf.duplicated(["city", "listing_id"]).any():
        raise AssertionError("gold_listing_features must have one row per (city, listing_id)")
    return pdf


def _replace_rows(sdf, table: str, cities: list, model_version: str):
    """Idempotent write: re-running replaces this model version's rows for these cities only."""
    if spark.catalog.tableExists(table):
        in_list = ", ".join(f"'{c}'" for c in cities)
        spark.sql(f"DELETE FROM {table} WHERE model_version = '{model_version}' AND city IN ({in_list})")
    sdf.write.format("delta").mode("append").saveAsTable(table)


def write_predictions(scored: pd.DataFrame, model_version: str):
    """All listings (also those without a price) -> gold_price_predictions(city, listing_id, predicted_price_eur, model_version)."""
    from pyspark.sql import functions as F
    out = scored[["city", "listing_id"]].copy()
    out["predicted_price_eur"] = scored["pred_model"].round(2).astype(float)
    out["model_version"] = model_version
    id_type = spark.table(GOLD_FEATURES).schema["listing_id"].dataType      # keep the key type of the feature table
    sdf = spark.createDataFrame(out).withColumn("listing_id", F.col("listing_id").cast(id_type))
    _replace_rows(sdf, GOLD_PREDICTIONS, sorted(out["city"].unique()), model_version)
    spark.sql(f"COMMENT ON TABLE {GOLD_PREDICTIONS} IS 'Predicted nightly price per listing from the price model. "
              f"One row per city, listing_id and model_version. Join to gold_listing_features on (city, listing_id). "
              f"Filter on one model_version before joining.'")
    for col, text in {
        "city": "City the listing belongs to (lower case).",
        "listing_id": "Airbnb listing id. Join key together with city.",
        "predicted_price_eur": "Price per night predicted by the model, in EUR. Compare with price_eur in gold_listing_features: predicted minus actual below zero means the model under-prices the listing.",
        "model_version": "Version of the model that made the prediction. v1 = trained on the training cities only.",
    }.items():
        safe = text.replace("'", "''")
        spark.sql(f"ALTER TABLE {GOLD_PREDICTIONS} ALTER COLUMN `{col}` COMMENT '{safe}'")
    print(f"{GOLD_PREDICTIONS}: wrote {len(out):,} rows for {sorted(out['city'].unique())}, model_version={model_version}")


def write_metrics(metrics: pd.DataFrame, model_version: str):
    """Error numbers -> gold_model_metrics, so the dashboard and the report read them instead of retyping them."""
    out = metrics.copy()
    out["model_version"] = model_version
    out["evaluated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    cols = ["model_version", "city", "evaluated_on", "segment", "n_listings", "mae_baseline_eur", "mae_model_eur",
            "improvement_pct", "median_ae_eur", "bias_eur", "median_price_eur", "evaluated_at"]
    schema = ("model_version string, city string, evaluated_on string, segment string, n_listings long, "
              "mae_baseline_eur double, mae_model_eur double, improvement_pct double, median_ae_eur double, "
              "bias_eur double, median_price_eur double, evaluated_at timestamp")
    out = out[cols].astype(object).where(out[cols].notna(), None)      # NaN -> NULL for Spark
    sdf = spark.createDataFrame(out, schema)
    _replace_rows(sdf, GOLD_MODEL_METRICS, sorted(out["city"].unique()), model_version)
    spark.sql(f"COMMENT ON TABLE {GOLD_MODEL_METRICS} IS 'Error of the price model and of the baseline (median price by "
              f"room type and guests) in EUR. evaluated_on = test (held-out 20 percent of a training city) or new_city "
              f"(all priced listings of a city the model never saw). segment = all, or one room type / stay type.'")
    print(f"{GOLD_MODEL_METRICS}: wrote {len(out):,} rows, model_version={model_version}")


def model_path(model_version: str) -> str:
    return f"{MODEL_DIR}/price_model_{model_version}.joblib"


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_artifact(artifact: dict, model_version: str) -> str:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{MODEL_VOLUME}")
    path = model_path(model_version)
    with tempfile.TemporaryDirectory() as tmp:               # write locally, then copy to the volume in one go
        local = os.path.join(tmp, "model.joblib")
        joblib.dump(artifact, local)
        shutil.copyfile(local, path)
    print(f"model saved: {path}\nsha256: {file_sha256(path)}")
    return path


def load_artifact(model_version: str) -> dict:
    path = model_path(model_version)
    artifact = joblib.load(path)
    print(f"model loaded: {path}\nsha256: {file_sha256(path)}")
    print(f"trained {artifact['trained_at']} on {artifact['train_cities']} | {artifact['model_name']}")
    if artifact["sklearn_version"] != sklearn.__version__:
        print(f"WARNING: saved with scikit-learn {artifact['sklearn_version']}, running {sklearn.__version__}")
    return artifact