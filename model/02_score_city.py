# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 · Score a new city with the saved model
# MAGIC
# MAGIC **Do not run this for city C before the repo has the tag `v1-AB`.**
# MAGIC
# MAGIC The city is a parameter. This notebook **fits nothing**: it loads the model saved by `06_train_evaluate`
# MAGIC and applies it as it is. So the first run on city C is the model exactly as it was for A and B.
# MAGIC
# MAGIC For a new city `is_test` is ignored: every listing is unseen, so the error is measured on all `is_trainable` rows.
# MAGIC
# MAGIC | Step | Output |
# MAGIC |---|---|
# MAGIC | 1 | Load the saved model (the sha256 shows it is the same file) |
# MAGIC | 2 | Did the data fit the model? Unseen categories, values outside the training range |
# MAGIC | 3 | The three numbers: baseline on the new city, model on the training cities, model on the new city |
# MAGIC | 4 | Why the gap: price level or ranking? Which listings? |
# MAGIC | 5 | Predictions for every listing to `gold_price_predictions` |

# COMMAND ----------

dbutils.widgets.text("cities", "munich", "City to score (comma separated for several)")
dbutils.widgets.text("model_version", "v1", "Model version to load")

# COMMAND ----------

# MAGIC %run ./00_model_common

# COMMAND ----------

CITIES = parse_cities(dbutils.widgets.get("cities"))
MODEL_VERSION = dbutils.widgets.get("model_version").strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Load the saved model and the new city

# COMMAND ----------

art = load_artifact(MODEL_VERSION)
FS = art["feature_set"]
overlap = sorted(set(CITIES) & set(art["train_cities"]))
if overlap:
    print(f"NOTE: {overlap} were training cities; their errors below are not 'new city' errors.")

raw = load_features(CITIES)
data = prepare(raw, art["property_types_kept"])
data = data.assign(pred_model=predict_price(art["pipeline"], FS, data),
                   pred_baseline=predict_baseline(art["baseline"], data))
priced = data[data["is_trainable"]].reset_index(drop=True)
print(f"{len(data):,} listings, {len(priced):,} with a usable price (is_trainable)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Did the data fit the model?
# MAGIC The model runs on anything, so "it ran" proves little. These checks show where it was guessing:
# MAGIC categories it never saw, and numeric values outside what it saw in training.

# COMMAND ----------

fit_rows = []
for c in [c for c in FEATURE_SETS[FS]["cat"] if c != "property_type"]:      # property_type has its own row below
    seen = set(art["train_categories"][c])
    unseen = ~data[c].isin(seen)
    examples = ", ".join(sorted(data.loc[unseen, c].dropna().astype(str).unique())[:5])
    fit_rows.append({"feature": c, "check": "category not seen in training",
                     "share_of_listings_pct": round(100 * unseen.mean(), 1), "detail": examples})
other = data["property_type"] == "Other"
fit_rows.append({"feature": "property_type", "check": "grouped into 'Other'",
                 "share_of_listings_pct": round(100 * other.mean(), 1),
                 "detail": ", ".join(data.loc[other, "property_type_raw"].value_counts().index[:5].astype(str))})
for c in FEATURE_SETS[FS]["num"]:
    lo, hi = art["train_ranges"][c]
    outside = (data[c] < lo) | (data[c] > hi)
    missing = data[c].isna()
    if outside.mean() >= 0.001 or missing.mean() > 0.05:
        fit_rows.append({"feature": c, "check": "outside training range / missing",
                         "share_of_listings_pct": round(100 * outside.mean(), 1),
                         "detail": f"train {lo:.2f} to {hi:.2f}; here {data[c].min():.2f} to {data[c].max():.2f}; "
                                   f"missing {100 * missing.mean():.1f}%"})
data_fit = pd.DataFrame(fit_rows).sort_values("share_of_listings_pct", ascending=False).reset_index(drop=True)
display(data_fit)

# COMMAND ----------

# MAGIC %md
# MAGIC Baseline fallbacks: how many listings have a room type and guest count that never appeared in train.

# COMMAND ----------

known = set(art["baseline"]["by_room_guests"])
fallback = ~pd.Series(list(zip(data["room_type"], data["accommodates"])), index=data.index).isin(known)
print(f"baseline used a fallback for {fallback.sum():,} of {len(data):,} listings ({100 * fallback.mean():.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · The three numbers, side by side

# COMMAND ----------

new_city = error_table(priced, "city", total_label="+".join(CITIES) if len(CITIES) > 1 else None)
trained = pd.DataFrame(art["test_metrics"])
side_by_side = pd.concat([trained.assign(evaluated_on="test (training city)"),
                          new_city.assign(evaluated_on="new city, all priced listings")], ignore_index=True)
display(side_by_side[["city", "evaluated_on", "n_listings", "median_price_eur",
                      "mae_baseline_eur", "mae_model_eur", "improvement_pct", "bias_eur"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Why the gap?
# MAGIC
# MAGIC **Price level or ranking.** If the new city is simply more expensive, the model under-prices almost every
# MAGIC listing by a similar factor (`bias_eur` far below zero). Rescaling the predictions by one number, the ratio of
# MAGIC median actual to median predicted price, removes that part. What is left is the error from ranking listings wrongly.
# MAGIC The rescaling uses the new city's prices, so it is a **diagnosis, not a result**: it is never written to the gold table.

# COMMAND ----------

decomp = []
for city, g in priced.groupby("city"):
    factor = g[TARGET].median() / np.median(g["pred_model"])
    mae = np.mean(np.abs(g["pred_model"] - g[TARGET]))
    mae_rescaled = np.mean(np.abs(g["pred_model"] * factor - g[TARGET]))
    decomp.append({
        "city": city,
        "median_price_train_eur": round(art["train_median_price"]["+".join(art["train_cities"])], 2),
        "median_price_here_eur": round(g[TARGET].median(), 2),
        "median_predicted_eur": round(float(np.median(g["pred_model"])), 2),
        "level_factor": round(factor, 3),
        "mae_model_eur": round(mae, 2),
        "mae_after_rescaling_eur": round(mae_rescaled, 2),
        "share_of_mae_from_price_level_pct": round(100 * (1 - mae_rescaled / mae), 1),
        "rank_correlation": round(g["pred_model"].corr(g[TARGET], method="spearman"), 3),
    })
display(pd.DataFrame(decomp))

# COMMAND ----------

# MAGIC %md
# MAGIC **Which listings.** Error by room type, stay type, size and distance to the centre.

# COMMAND ----------

priced = priced.assign(
    stay_type=np.where(priced["is_long_stay"] == 1.0, "long stay (30+ nights)", "short stay"),
    guests=pd.cut(priced["accommodates"], [0, 2, 4, 6, 100], labels=["1-2", "3-4", "5-6", "7+"]).astype(str),
    distance=pd.cut(priced["dist_to_centre_km"], [-0.01, 2, 5, 10, 1e6],
                    labels=["a) 0-2 km", "b) 2-5 km", "c) 5-10 km", "d) 10+ km"]).astype(str),
)
for key in ["room_type", "stay_type", "guests", "distance"]:
    display(error_table(priced, key))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Write predictions and metrics
# MAGIC Every listing of the city, also those without a price.

# COMMAND ----------

assert np.isfinite(data["pred_model"]).all() and (data["pred_model"] > 0).all()
write_predictions(data, MODEL_VERSION)

metrics = pd.concat(
    [new_city.assign(segment="all")]
    + [error_table(g, key).rename(columns={key: "segment"}).assign(city=city)
       for city, g in priced.groupby("city") for key in ["room_type", "stay_type"]],
    ignore_index=True).assign(evaluated_on="new_city")
write_metrics(metrics, MODEL_VERSION)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary for the report

# COMMAND ----------

ab = trained.iloc[-1]
print(f"Model {MODEL_VERSION} ({art['model_name']}), trained {art['trained_at']} on {art['train_cities']}, applied unchanged\n")
print(f"model on {ab['city']} (test): MAE {ab['mae_model_eur']:.2f} EUR")
for r in new_city.to_dict("records"):
    print(f"baseline on {r['city']}: MAE {r['mae_baseline_eur']:.2f} EUR")
    print(f"model on {r['city']}:    MAE {r['mae_model_eur']:.2f} EUR  (bias {r['bias_eur']:+.2f} EUR, n = {r['n_listings']:,})")