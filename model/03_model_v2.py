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
# MAGIC
# MAGIC The changes are applied one on top of the other, so the table at the end shows what each one buys.
# MAGIC
# MAGIC **Model v2 is step B** (parameter `final_step`). The assignment reports the error on A and B and on city C
# MAGIC separately, also after the changes, so city C stays a city the model is **not trained on**. Both steps fit the
# MAGIC model on Berlin and Vienna rows only. In step B Munich contributes one number, its median price, which is used
# MAGIC when a prediction is converted to euros, not in training. No model here is trained on Munich listings.
# MAGIC
# MAGIC **How the comparison stays honest**
# MAGIC * B takes Munich's median price from Munich's `is_test = false` rows, so it is measured only on Munich's held-out rows (`is_test = true`).
# MAGIC * v1 and the baseline are measured again on those same rows, so every row of the table is the same listings.
# MAGIC * For reference the table also has a baseline built from Munich's own medians. It is not a model and is not used
# MAGIC   for predictions; it shows how hard Munich prices are to predict even with Munich data.
# MAGIC * Same algorithm and settings as v1. Nothing is tuned on Munich.

# COMMAND ----------

dbutils.widgets.text("train_cities_v1", "berlin,vienna", "Training cities of v1")
dbutils.widgets.text("new_city", "munich", "New city")
dbutils.widgets.text("base_version", "v1", "Model version of the first run")
dbutils.widgets.text("new_version", "v2", "Model version to create")
dbutils.widgets.dropdown("final_step", "B", ["A", "B"], "Step saved as the new version")

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
FINAL_STEP = dbutils.widgets.get("final_step").strip().upper()
assert FINAL_STEP in ("A", "B")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Data and the v1 model

# COMMAND ----------

v1 = load_artifact(BASE_VERSION)
raw = load_features(ALL_CITIES)


def split_rows(data: pd.DataFrame):
    trainable = data[data["is_trainable"]]
    return trainable[~trainable["is_test"]].reset_index(drop=True), trainable[trainable["is_test"]].reset_index(drop=True)


# Property types are the ones v1 knew (learned on Berlin + Vienna train rows).
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
baseline_new = fit_baseline(train_new)           # reference only: medians of Munich's is_test = false rows

# --- A: portable features, Berlin + Vienna data only ------------------------------------------
model_a = fit_model(make_gbm, FS_V2, train_old)

# --- B: A + relative price; Munich contributes one number, its median train price -------------
levels_b = city_levels(train)                    # Berlin, Vienna and the Munich level (is_test = false rows)
model_b = fit_relative(train_old, levels_b, FS_V2)

print("city price levels (median price of the train rows, EUR):", {c: round(v, 2) for c, v in levels_b.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · The result — same held-out listings in every row
# MAGIC `mae_new_city_test` is measured on Munich's held-out rows, `mae_old_cities_test` on the Berlin + Vienna test set.
# MAGIC The last column shows whether a change that helps Munich costs anything in the cities we already had.

# COMMAND ----------

ladder = pd.DataFrame([
    {"step": f"baseline, {AB} medians", "munich_prices_used": "none",
     "mae_new_city_test": mae(predict_baseline(baseline_old, test_new), test_new),
     "mae_old_cities_test": mae(predict_baseline(baseline_old, test_old), test_old)},
    {"step": f"baseline, {NEW_CITY} medians (reference)", "munich_prices_used": "medians of is_test = false rows",
     "mae_new_city_test": mae(predict_baseline(baseline_new, test_new), test_new), "mae_old_cities_test": None},
    {"step": f"{BASE_VERSION}: first run, unchanged", "munich_prices_used": "none",
     "mae_new_city_test": mae(predict_price(v1["pipeline"], v1["feature_set"], test_new), test_new),
     "mae_old_cities_test": mae(predict_price(v1["pipeline"], v1["feature_set"], test_old), test_old)},
    {"step": "A: without city, latitude, longitude", "munich_prices_used": "none",
     "mae_new_city_test": mae(predict_price(model_a, FS_V2, test_new), test_new),
     "mae_old_cities_test": mae(predict_price(model_a, FS_V2, test_old), test_old)},
    {"step": "B: A + price relative to the city level", "munich_prices_used": "one number (median price of is_test = false rows)",
     "mae_new_city_test": mae(predict_relative(model_b, FS_V2, test_new, levels_b), test_new),
     "mae_old_cities_test": mae(predict_relative(model_b, FS_V2, test_old, levels_b), test_old)},
])
ladder.insert(2, "n_new_city_test", len(test_new))
ladder["saved_as"] = np.where(ladder["step"].str.startswith(FINAL_STEP + ":"), NEW_VERSION, "")
display(ladder)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Model v2 (the step chosen in `final_step`) in detail
# MAGIC Error per city on the held-out rows, against the baseline from Berlin + Vienna medians (the same baseline as in the first run).

# COMMAND ----------

STEPS = {
    "A": dict(pipe=model_a, levels=None, name=f"gbm | {FS_V2}"),
    "B": dict(pipe=model_b, levels=levels_b, name=f"gbm | {FS_V2} | relative to city level"),
}
final = STEPS[FINAL_STEP]


def predict_final(rows: pd.DataFrame) -> np.ndarray:
    if final["levels"] is None:
        return predict_price(final["pipe"], FS_V2, rows)
    return predict_relative(final["pipe"], FS_V2, rows, final["levels"])


test_f = test.assign(
    pred_model=predict_final(test),
    stay_type=np.where(test["is_long_stay"] == 1.0, "long stay (30+ nights)", "short stay"),
    guests=pd.cut(test["accommodates"], [0, 2, 4, 6, 100], labels=["1-2", "3-4", "5-6", "7+"]).astype(str),
)
test_f["pred_baseline"] = predict_baseline(baseline_old, test_f)
print(f"{NEW_VERSION} = step {FINAL_STEP}: {final['name']}, trained on {OLD_CITIES} only")
headline = error_table(test_f, "city")
display(headline)

# COMMAND ----------

new_rows = test_f[test_f["city"] == NEW_CITY]
print(f"{NEW_CITY}: rank correlation of predicted and actual price "
      f"{new_rows['pred_model'].corr(new_rows[TARGET], method='spearman'):.3f}")
for key in ["room_type", "stay_type", "guests"]:
    display(error_table(new_rows, key))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Save v2, write predictions and metrics
# MAGIC Predictions for every listing of the three cities go to `gold_price_predictions` with `model_version = v2`.
# MAGIC The v1 rows stay, so the dashboard can show the first run next to the result after the changes.
# MAGIC Re-running replaces the earlier v2 rows.

# COMMAND ----------

artifact = {
    "model_version": NEW_VERSION,
    "final_step": FINAL_STEP,
    "model_name": final["name"],
    "feature_set": FS_V2,
    "features": FEATURE_SETS[FS_V2],
    "pipeline": final["pipe"],
    "city_levels": final["levels"],
    "baseline": baseline_old,
    "property_types_kept": v1["property_types_kept"],
    "train_cities": OLD_CITIES,
    "ladder": ladder.to_dict("records"),
    "test_metrics": headline.to_dict("records"),
    "n_train": int(len(train_old)),
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "sklearn_version": sklearn.__version__,
}
save_artifact(artifact, NEW_VERSION)

# COMMAND ----------

scored = data.assign(pred_model=predict_final(data))
assert np.isfinite(scored["pred_model"]).all() and (scored["pred_model"] > 0).all()
write_predictions(scored, NEW_VERSION)

# the model is not trained on the new city, so its rows are "new_city", not "test"
metrics = pd.concat(
    [headline.assign(segment="all")]
    + [error_table(g, key).rename(columns={key: "segment"}).assign(city=city)
       for city, g in test_f.groupby("city") for key in ["room_type", "stay_type"]],
    ignore_index=True)
metrics["evaluated_on"] = np.where(metrics["city"] == NEW_CITY, "new_city", "test")
write_metrics(metrics, NEW_VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary for the report

# COMMAND ----------

print(f"First run, {BASE_VERSION} on all priced {NEW_CITY} listings: MAE {first_run_mae} EUR\n")
print(f"On the {len(test_new):,} held-out {NEW_CITY} listings:")
for r in ladder.to_dict("records"):
    old = "" if pd.isna(r["mae_old_cities_test"]) else f"   ({AB} test: {r['mae_old_cities_test']:.2f})"
    print(f"  {r['step']:<45} {r['mae_new_city_test']:7.2f} EUR{old}{'   <-- ' + NEW_VERSION if r['saved_as'] else ''}")