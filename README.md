# strudel-team_Germany_Austria_Assignment_2

## Data

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/)

| Role | City | Snapshot date |
|---|---|---|
| A (training) | Berlin | 2026-09-19 |
| B (training) | Vienna | 2026-09-17 |
| C (new client) | Munich | — not loaded until the A/B pipeline and model are finished |

Raw files are stored in a Databricks Volume (not in git):
`/Volumes/workspace/airbnb/raw/<city>/listings.csv.gz`

## Exploratory analysis — `eda/`

`eda/eda_berlin_vienna.ipynb` — EDA on Berlin and Vienna:
- dataset description, missing values and outliers;
- what drives price;
- cleaning rules for the pipeline;
- leakage review (which columns must not be used to predict price).
