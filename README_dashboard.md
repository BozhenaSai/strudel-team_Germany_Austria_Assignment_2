# AI/BI dashboard

`airbnb_prices.lvdash.json` is the whole dashboard: datasets, pages, widgets and filters. `datasets.sql` holds the SQL of every dataset, for reading and review. Every dataset reads only gold tables in `workspace.airbnb` (including `gold_baseline_predictions` from `baseline/01_baseline`) and keeps the latest snapshot of each city.

**To open it:** Dashboards → arrow next to **Create dashboard** → **Import dashboard from file** → choose `airbnb_prices.lvdash.json`. Then pick the serverless warehouse and press **Publish**.

Every page title states a finding (Lecture 7).

## Page 1. City overview

*Munich is 60 percent more expensive than Berlin and Vienna, which cost almost the same.*

City profile table (listings, median price, availability, review score, hosts, long stays, superhosts, listings without price), room type mix, median price by room type, price and availability distributions, the top 10 neighbourhoods by median price and by number of listings. Filter: city.

## Page 2. Berlin vs Vienna

*Berlin and Vienna cost the same per night, but Berlin is a long-stay market and Vienna a professional short-stay one.*

Side-by-side metrics with the difference, median price by room type, by stay type and by distance to the centre, room type mix, and a text box on what matters for the model: stay type, missing prices, location.

## Page 3. Model

*Trained only on Berlin and Vienna, the model beats the baseline by 31 percent there; in Munich v2 cuts the error from 157 to 115 EUR.*

Both models are trained on Berlin and Vienna only; no model is trained on Munich. v1 is the model of the first run, v2 the model after the changes.

Which listings each number covers:

- Berlin and Vienna: the 20 percent test listings, for every version.
- Munich, v1: all 5,076 priced listings (the first run used nothing from Munich).
- Munich, v2: only the 1,019 held-out listings (`is_test`), because v2 takes Munich's median price from the other Munich listings.

Widgets: MAE of baseline and model per city; **Munich, v1 against v2 on the same 1,019 held-out listings** (baseline 171.31, v1 156.90, v2 115.36 EUR) with a table per neighbourhood; predicted against actual price; MAE by room type; average error by stay type; the neighbourhoods the model under-prices most; the 20 most under-priced and over-priced listings per city (in EUR, with a link); a text box on the Munich first run and the changes.

Filters: **Model version** (default `v1`) and **City**. The Munich comparison widgets do not depend on the version filter, they always show baseline, v1 and v2 together.

Error = predicted minus actual price, so below zero means under-priced.
