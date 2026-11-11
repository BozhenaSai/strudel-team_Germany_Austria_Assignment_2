# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Model v2 — the changes after the first run on city C
# MAGIC
# MAGIC **First run (model v1, unchanged):** MAE on Munich 149.61 EUR, against 40.39 EUR on Berlin + Vienna.
# MAGIC The pipeline needed no fixes (participant 2 checked encoding, types, outliers, missing values, row balance,
# MAGIC city centre and feature distributions), so everything below is a **model change**.
# MAGIC
# MAGIC | # | Model change | Reason (from the first run) | Needs Munich prices? |
# MAGIC |---|---|---|---|
# MAGIC | A | Remove `city`, `latitude`, `longitude` | `city = munich` was never seen; every Munich longitude is outside the training range. `dist_to_centre_km` carries location the same way in every city. | No |
# MAGIC | B | Predict the price **relative to the city's median price**, then multiply by the level of the city being scored | A price level the model never saw: Munich's median is 204 EUR against about 126 EUR in training. | Yes: one number, the median of Munich's train rows |
# MAGIC | C | Retrain on Berlin + Vienna + Munich train rows | One factor fixed only a quarter of the error: Munich prices are also spread wider (size and centre cost more). The model has to see Munich to learn that. | Yes: Munich's train rows |
# MAGIC
# MAGIC The changes are applied one on top of the other, so the table at the end shows what each one buys.
# MAGIC
# MAGIC **How the comparison stays honest**
# MAGIC * B and C use Munich prices, so they are measured only on Munich's held-out rows (`is_test = true`), which they never see.
# MAGIC * v1 and the baseline are measured again on those same rows, so every row of the table is the same listings.
# MAGIC * A model that has seen Munich is also compared with a baseline that has seen Munich (medians of Munich's own train rows).
# MAGIC * Same algorithm and settings as v1. Nothing is tuned on Munich.

# COMMAND ----------

dbutils.widgets.text("train_cities_v1", "berlin,vienna", "Training cities of v1")
dbutils.widgets.text("new_city", "munich", "New city")
dbutils.widgets.text("base_version", "v1", "Model version of the first run")
dbutils.widgets.text("new_version", "v2", "Model version to create")

# COMMAND ----------

# MAGIC %run ./00_model_common

# COMMAND ----------

OLD_CITIES = parse_cities(dbutils.widgets.get("train_cities_v1"))
NEW_CITY = parse_cities(dbutils.widgets.get("new_city"))[0]
BASE_VERSION = dbutils.widgets.get("base_version").strip()
NEW_VERSION = dbutils.widgets.get("new_version").strip()
ALL_CITIES = OLD_CITIES + [NEW_CITY]
FS_V2 = "core"                      # change A: the portable feature set, no city, no raw coordinates
AB = "+".join(OLD_CITIES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Data and the v1 model

# COMMAND ----------

v1 = load_artifact(BASE_VERSION)
raw = load_features(ALL_CITIES)


def split_rows(data: pd.DataFrame):
    trainable = data[data["is_trainable"]]
    return trainable[~trainable["is_test"]].reset_index(drop=True), trainable[trainable["is_test"]].reset_index(drop=True)


# For v1, A and B the property types are the ones v1 knew (learned on Berlin + Vienna train rows).
data = prepare(raw, v1["property_types_kept"])
train, test = split_rows(data)
train_old = train[train["city"].isin(OLD_CITIES)].reset_index(drop=True)
train_new = train[train["city"] == NEW_CITY].reset_index(drop=True)
test_old = test[test["city"].isin(OLD_CITIES)].reset_index(drop=True)
test_new = test[test["city"] == NEW_CITY].reset_index(drop=True)
all_new = data[(data["city"] == NEW_CITY) & data["is_trainable"]].reset_index(drop=True)

display(pd.DataFrame([{"city": c, "train_rows": int((train["city"] == c).sum()), "test_rows": int((test["city"] == c).sum())}
                      for c in ALL_CITIES]))


def mae(pred, rows) -> float:
    return round(float(np.mean(np.abs(np.asarray(pred) - rows[TARGET].to_numpy()))), 2)


# consistency check: v1 on every priced Munich listing must reproduce the first-run number
first_run_mae = mae(predict_price(v1["pipeline"], v1["feature_set"], all_new), all_new)
print(f"v1 on all {len(all_new):,} priced {NEW_CITY} listings: MAE {first_run_mae} EUR (first run)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · The changes, one on top of the other
# MAGIC
# MAGIC **Relative price (change B).** The model learns `log(price / city level)`, where the city level is the median
# MAGIC price of the city's train rows. A prediction is the relative price times the level of the city being scored.
# MAGIC The model then learns what makes a listing dearer or cheaper *than is usual in its city*, and one number per
# MAGIC city carries the price level.

# COMMAND ----------

def city_levels(train_rows: pd.DataFrame) -> dict:
    return {c: float(v) for c, v in train_rows.groupby("city")[TARGET].median().items()}


def fit_relative(train_rows: pd.DataFrame, levels: dict, fs: str):
    pipe = make_gbm(fs)
    pipe.fit(train_rows[feature_columns(fs)], np.log(train_rows[TARGET] / train_rows["city"].map(levels)))
    return pipe


def predict_relative(pipe, fs: str, rows: pd.DataFrame, levels: dict) -> np.ndarray:
    return np.exp(pipe.predict(rows[feature_columns(fs)])) * rows["city"].map(levels).to_numpy()


# --- baselines -------------------------------------------------------------------------------
baseline_old = v1["baseline"]                    # medians of Berlin + Vienna train rows (the first-run baseline)
baseline_new = fit_baseline(train_new)           # medians of Munich's own train rows

# --- A: portable features, Berlin + Vienna data only ------------------------------------------
model_a = fit_model(make_gbm, FS_V2, train_old)

# --- B: A + relative price; Munich contributes one number, its median train price -------------
levels_b = city_levels(train)                    # includes the Munich level
model_b = fit_relative(train_old, levels_b, FS_V2)

# --- C: B retrained on the train rows of all three cities -------------------------------------
property_types_kept = learn_property_types(raw[raw["is_trainable"].fillna(False).astype(bool)
                                               & ~raw["is_test"].fillna(False).astype(bool)])
data_c = prepare(raw, property_types_kept)
train_c, test_c = split_rows(data_c)
levels_c = city_levels(train_c)
model_c = fit_relative(train_c, levels_c, FS_V2)

print("city price levels (median price of the train rows, EUR):", {c: round(v, 2) for c, v in levels_c.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · The result — same held-out listings in every row
# MAGIC `mae_new_city_test` is measured on Munich's held-out rows, `mae_old_cities_test` on the Berlin + Vienna test set.
# MAGIC The last column shows whether a change that helps Munich costs anything in the cities we already had.

# COMMAND ----------

test_new_c = test_c[test_c["city"] == NEW_CITY].reset_index(drop=True)
test_old_c = test_c[test_c["city"].isin(OLD_CITIES)].reset_index(drop=True)
assert (test_new_c["listing_id"].to_numpy() == test_new["listing_id"].to_numpy()).all()

ladder = pd.DataFrame([
    {"step": f"baseline, {AB} medians", "munich_prices_used": "none",
     "mae_new_city_test": mae(predict_baseline(baseline_old, test_new), test_new),
     "mae_old_cities_test": mae(predict_baseline(baseline_old, test_old), test_old)},
    {"step": f"baseline, {NEW_CITY} medians", "munich_prices_used": "train rows",
     "mae_new_city_test": mae(predict_baseline(baseline_new, test_new), test_new), "mae_old_cities_test": None},
    {"step": f"{BASE_VERSION}: first run, unchanged", "munich_prices_used": "none",
     "mae_new_city_test": mae(predict_price(v1["pipeline"], v1["feature_set"], test_new), test_new),
     "mae_old_cities_test": mae(predict_price(v1["pipeline"], v1["feature_set"], test_old), test_old)},
    {"step": "A: without city, latitude, longitude", "munich_prices_used": "none",
     "mae_new_city_test": mae(predict_price(model_a, FS_V2, test_new), test_new),
     "mae_old_cities_test": mae(predict_price(model_a, FS_V2, test_old), test_old)},
    {"step": "B: A + price relative to the city level", "munich_prices_used": "one number (median of train rows)",
     "mae_new_city_test": mae(predict_relative(model_b, FS_V2, test_new, levels_b), test_new),
     "mae_old_cities_test": mae(predict_relative(model_b, FS_V2, test_old, levels_b), test_old)},
    {"step": f"C: B retrained with {NEW_CITY} = {NEW_VERSION}", "munich_prices_used": "train rows",
     "mae_new_city_test": mae(predict_relative(model_c, FS_V2, test_new_c, levels_c), test_new_c),
     "mae_old_cities_test": mae(predict_relative(model_c, FS_V2, test_old_c, levels_c), test_old_c)},
])
ladder.insert(2, "n_new_city_test", len(test_new))
display(ladder)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Model v2 (step C) in detail
# MAGIC Test-set error per city. For Munich the baseline is the median of Munich's own train rows, because v2 has
# MAGIC seen Munich; for Berlin and Vienna it is the same baseline as in v1.

# COMMAND ----------

test_c = test_c.assign(
    pred_model=predict_relative(model_c, FS_V2, test_c, levels_c),
    stay_type=np.where(test_c["is_long_stay"] == 1.0, "long stay (30+ nights)", "short stay"),
    guests=pd.cut(test_c["accommodates"], [0, 2, 4, 6, 100], labels=["1-2", "3-4", "5-6", "7+"]).astype(str),
)
test_c["pred_baseline"] = np.where(test_c["city"] == NEW_CITY,
                                   predict_baseline(baseline_new, test_c), predict_baseline(baseline_old, test_c))
headline = error_table(test_c, "city")
display(headline)

# COMMAND ----------

new_rows = test_c[test_c["city"] == NEW_CITY]
print(f"{NEW_CITY}: rank correlation of predicted and actual price "
      f"{new_rows['pred_model'].corr(new_rows[TARGET], method='spearman'):.3f}")
for key in ["room_type", "stay_type", "guests"]:
    display(error_table(new_rows, key))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Save v2, write predictions and metrics
# MAGIC Predictions for every listing of the three cities go to `gold_price_predictions` with `model_version = v2`.
# MAGIC The v1 rows stay, so the dashboard can show the first run next to the result after the changes.

# COMMAND ----------

artifact = {
    "model_version": NEW_VERSION,
    "model_name": f"gbm | {FS_V2} | relative to city level",
    "feature_set": FS_V2,
    "features": FEATURE_SETS[FS_V2],
    "pipeline": model_c,
    "city_levels": levels_c,
    "baseline": baseline_old,
    "baseline_new_city": baseline_new,
    "property_types_kept": property_types_kept,
    "train_cities": ALL_CITIES,
    "ladder": ladder.to_dict("records"),
    "test_metrics": headline.to_dict("records"),
    "n_train": int(len(train_c)),
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "sklearn_version": sklearn.__version__,
}
save_artifact(artifact, NEW_VERSION)

# COMMAND ----------

scored = data_c.assign(pred_model=predict_relative(model_c, FS_V2, data_c, levels_c))
assert np.isfinite(scored["pred_model"]).all() and (scored["pred_model"] > 0).all()
write_predictions(scored, NEW_VERSION)

metrics = pd.concat(
    [headline.assign(segment="all")]
    + [error_table(g, key).rename(columns={key: "segment"}).assign(city=city)
       for city, g in test_c.groupby("city") for key in ["room_type", "stay_type"]],
    ignore_index=True).assign(evaluated_on="test")
write_metrics(metrics, NEW_VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary for the report

# COMMAND ----------

print(f"First run, {BASE_VERSION} on all priced {NEW_CITY} listings: MAE {first_run_mae} EUR\n")
print(f"On the {len(test_new):,} held-out {NEW_CITY} listings:")
for r in ladder.to_dict("records"):
    old = "" if pd.isna(r["mae_old_cities_test"]) else f"   ({AB} test: {r['mae_old_cities_test']:.2f})"
    print(f"  {r['step']:<45} {r['mae_new_city_test']:7.2f} EUR{old}")