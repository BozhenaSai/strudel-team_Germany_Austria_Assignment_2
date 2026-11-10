# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Price model — train and evaluate on the training cities
# MAGIC
# MAGIC **Team Shtrudel · A = Berlin, B = Vienna.** City C is not touched here.
# MAGIC
# MAGIC | Step | What happens | Which rows |
# MAGIC |---|---|---|
# MAGIC | 1 | Load `gold_listing_features`, check the split | all |
# MAGIC | 2 | Choose the model by 5-fold cross-validation | **train only** (`is_trainable` and not `is_test`) |
# MAGIC | 3 | Fit the chosen model once | train only |
# MAGIC | 4 | Report MAE against the baseline | **test only** (`is_trainable` and `is_test`), used once |
# MAGIC | 5 | Explain the model: which features matter | test |
# MAGIC | 6 | Save the model, write predictions for every listing | all |
# MAGIC
# MAGIC The test set is fixed by the pipeline before any model exists, and the choice in step 2 never sees it.
# MAGIC Every number we report comes from step 4.

# COMMAND ----------

dbutils.widgets.text("train_cities", "berlin,vienna", "Training cities (comma separated)")
dbutils.widgets.text("model_version", "v1", "Model version")

# COMMAND ----------

# MAGIC %run ./00_model_common

# COMMAND ----------

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

TRAIN_CITIES = parse_cities(dbutils.widgets.get("train_cities"))
MODEL_VERSION = dbutils.widgets.get("model_version").strip()
CV_FOLDS = 5
TOLERANCE_PCT = 1.0      # a simpler candidate wins if its CV MAE is within this many percent of the best one
print(f"train cities: {TRAIN_CITIES}   model_version: {MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Data and split

# COMMAND ----------

raw = load_features(TRAIN_CITIES)

trainable_raw = raw[raw["is_trainable"].fillna(False).astype(bool)]
is_test_raw = trainable_raw["is_test"].fillna(False).astype(bool)
property_types_kept = learn_property_types(trainable_raw[~is_test_raw])      # learned on train rows only

data = prepare(raw, property_types_kept)
trainable = data[data["is_trainable"]]
train = trainable[~trainable["is_test"]].reset_index(drop=True)
test = trainable[trainable["is_test"]].reset_index(drop=True)

assert train[TARGET].notna().all() and (train[TARGET] > 0).all(), "trainable rows must have a positive price"
assert not set(zip(train.city, train.listing_id)) & set(zip(test.city, test.listing_id)), "train and test overlap"

split = (data.assign(part=np.select([~data["is_trainable"], data["is_test"]], ["not trainable", "test"], "train"))
         .groupby(["city", "part"]).size().unstack(fill_value=0))
split["test_share_pct"] = (100 * split["test"] / (split["test"] + split["train"])).round(1)
display(split.reset_index())
print(f"property types kept as their own category: {len(property_types_kept)} "
      f"(the other {raw['property_type'].nunique() - len(property_types_kept)} become 'Other')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Model choice — cross-validation on the train rows only
# MAGIC
# MAGIC Five folds, stratified by city so every fold has both cities in the same proportion. For each candidate we
# MAGIC predict the held-out fold and measure MAE in EUR. The baseline goes through the same folds.
# MAGIC
# MAGIC **Rule, fixed before looking at the numbers:** take the lowest cross-validated MAE, but if a simpler or more
# MAGIC portable candidate (earlier in the list) is within 1% of it, take that one. A feature that buys nothing
# MAGIC measurable is not worth explaining.

# COMMAND ----------

def cross_validate(train: pd.DataFrame) -> pd.DataFrame:
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(train, train["city"]))
    oof = {name: np.zeros(len(train)) for name, _, _ in CANDIDATES}
    oof["baseline"] = np.zeros(len(train))
    for fit_idx, val_idx in folds:
        fit_part, val_part = train.iloc[fit_idx], train.iloc[val_idx]
        oof["baseline"][val_idx] = predict_baseline(fit_baseline(fit_part), val_part)
        for name, maker, fs in CANDIDATES:
            oof[name][val_idx] = predict_price(fit_model(maker, fs, fit_part), fs, val_part)
    rows = []
    for name, pred in oof.items():
        err = np.abs(pred - train[TARGET].to_numpy())
        row = {"candidate": name, "cv_mae_eur": err.mean()}
        for city in sorted(train["city"].unique()):
            row[f"cv_mae_{city}"] = err[(train["city"] == city).to_numpy()].mean()
        rows.append(row)
    return pd.DataFrame(rows).round(2)


cv = cross_validate(train)

order = [name for name, _, _ in CANDIDATES]
best = cv[cv["candidate"].isin(order)]["cv_mae_eur"].min()
within = cv[(cv["candidate"].isin(order)) & (cv["cv_mae_eur"] <= best * (1 + TOLERANCE_PCT / 100))]["candidate"].tolist()
CHOSEN = next(name for name in order if name in within)
_, CHOSEN_MAKER, CHOSEN_FS = next(c for c in CANDIDATES if c[0] == CHOSEN)

cv["chosen"] = np.where(cv["candidate"] == CHOSEN, "<-- chosen", "")
display(cv)
print(f"lowest CV MAE: {best:.2f} EUR   chosen: {CHOSEN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Fit the chosen model on all train rows

# COMMAND ----------

model = fit_model(CHOSEN_MAKER, CHOSEN_FS, train)
baseline = fit_baseline(train)
print(f"fitted {CHOSEN} on {len(train):,} train listings, {len(feature_columns(CHOSEN_FS))} features:")
print(", ".join(feature_columns(CHOSEN_FS)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Test-set result — the numbers we report
# MAGIC MAE in EUR on the held-out test listings, for each city and both together, against the baseline.

# COMMAND ----------

test = test.assign(pred_model=predict_price(model, CHOSEN_FS, test), pred_baseline=predict_baseline(baseline, test))
BOTH = "+".join(TRAIN_CITIES)

headline = error_table(test, "city", total_label=BOTH)
display(headline)

# COMMAND ----------

# MAGIC %md
# MAGIC Where the error sits. Long-stay listings are reported separately, as promised in the EDA: their nightly
# MAGIC price is derived from a monthly price, so they are a different product.

# COMMAND ----------

test = test.assign(stay_type=np.where(test["is_long_stay"] == 1.0, "long stay (30+ nights)", "short stay"),
                   guests=pd.cut(test["accommodates"], [0, 2, 4, 6, 100], labels=["1-2", "3-4", "5-6", "7+"]).astype(str))
display(error_table(test, "room_type"))
display(error_table(test, "stay_type"))
display(error_table(test, "guests"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · What the model uses
# MAGIC Permutation importance on the test set: shuffle one feature, see how much the MAE (in EUR) gets worse.
# MAGIC It needs no knowledge of the model's internals and reads directly as "this feature is worth X EUR of error".

# COMMAND ----------

eur_mae = make_scorer(lambda y_log, p_log: np.mean(np.abs(np.exp(p_log) - np.exp(y_log))), greater_is_better=False)
cols = feature_columns(CHOSEN_FS)
imp = permutation_importance(model, test[cols], np.log(test[TARGET]), scoring=eur_mae, n_repeats=5, random_state=SEED)
importance = (pd.DataFrame({"feature": cols, "mae_increase_eur": imp.importances_mean.round(2)})
              .sort_values("mae_increase_eur", ascending=False).reset_index(drop=True))
display(importance)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Save the model and write predictions
# MAGIC
# MAGIC The saved file holds everything `02_score_city` needs: the fitted model, its feature list, the property types
# MAGIC it knows, the baseline medians and the range of every numeric feature in train. Scoring a new city loads this
# MAGIC file and fits nothing. The printed sha256 identifies this exact model; note it next to the `v1-AB` tag.

# COMMAND ----------

artifact = {
    "model_version": MODEL_VERSION,
    "model_name": CHOSEN,
    "feature_set": CHOSEN_FS,
    "pipeline": model,
    "baseline": baseline,
    "property_types_kept": property_types_kept,
    "train_cities": TRAIN_CITIES,
    "train_categories": {c: sorted(train[c].dropna().unique().tolist()) for c in FEATURE_SETS[CHOSEN_FS]["cat"]},
    "train_ranges": {c: (float(train[c].min()), float(train[c].max())) for c in FEATURE_SETS[CHOSEN_FS]["num"]},
    "train_median_price": {**train.groupby("city")[TARGET].median().to_dict(), BOTH: float(train[TARGET].median())},
    "test_metrics": headline.to_dict("records"),
    "n_train": int(len(train)),
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "sklearn_version": sklearn.__version__,
}
save_artifact(artifact, MODEL_VERSION)

# COMMAND ----------

# every listing of the training cities, also those without a price and the outliers
scored = data.assign(pred_model=predict_price(model, CHOSEN_FS, data))
assert np.isfinite(scored["pred_model"]).all() and (scored["pred_model"] > 0).all()
write_predictions(scored, MODEL_VERSION)

# COMMAND ----------

metrics = pd.concat(
    [headline.assign(segment="all")]
    + [error_table(g, key).rename(columns={key: "segment"}).assign(city=city)
       for city, g in list(test.groupby("city")) + [(BOTH, test)] for key in ["room_type", "stay_type"]],
    ignore_index=True).assign(evaluated_on="test")
write_metrics(metrics, MODEL_VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary for the report

# COMMAND ----------

print(f"Model {MODEL_VERSION}: {CHOSEN}, trained on {len(train):,} listings, tested on {len(test):,}\n")
for r in headline.to_dict("records"):
    print(f"{r['city']:>16}: baseline {r['mae_baseline_eur']:6.2f} EUR | model {r['mae_model_eur']:6.2f} EUR "
          f"| {r['improvement_pct']:.1f}% lower | n = {r['n_listings']:,}")