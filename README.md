# Global Inflation Analysis (1970–2022)

Exploratory data analysis and time series modelling of headline CPI inflation across 212 countries using IMF/World Bank data.

## What this project covers

- **EDA & descriptive statistics** — global inflation by decade, distribution by regime
- **Time series visualisation** — Brazil, USA, Germany, Argentina, Japan, China (1970–2022)
- **Inflation regimes** — oil shocks, Volcker disinflation, GFC, post-COVID surge
- **Cross-country correlation** — Pearson correlation matrix and rolling 10-year correlations
- **ARIMA forecast** — stationarity test (ADF), ACF/PACF, ARIMA(1,1,1) for Brazil with 90% CI

## Figures produced

| Figure | Description |
|---|---|
| `01_inflation_selected_countries.png` | Time series for 6 countries with regime shading |
| `02_world_inflation_regimes.png` | Global median ± IQR with annotated periods |
| `03_correlation_heatmap.png` | Pairwise Pearson correlation across 6 countries |
| `04_rolling_correlation_bra_usa.png` | 10-year rolling correlation: Brazil vs USA |
| `05_acf_pacf_brazil.png` | ACF/PACF diagnostics for ARIMA selection |
| `06_arima_forecast_brazil.png` | ARIMA(1,1,1) forecast for Brazil 2023–2027 |

## Data

**Source:** Global Inflation Dataset — 212 countries, 1970–2022 (IMF/World Bank), via Kaggle.

Download with:
```bash
# Requires a Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d belayethossainds/global-inflation-dataset-212-country-19702022 \
  --path data/raw --unzip
```

Raw data is excluded from version control (see `.gitignore`). The notebook is fully reproducible once the CSV is placed in `data/raw/`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_inflation_global_analysis.ipynb
```

## Stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `statsmodels`

## Key findings

- Global inflation peaked in the 1970s–80s (oil shocks); the world median exceeded 10% from 1973–82.
- The 1990s disinflation was synchronised across advanced and emerging economies.
- The 2021–22 post-COVID surge was the first globally synchronised inflation spike since the 1970s.
- Brazil's modern inflation regime (post-1995 Real Plan) is mean-reverting; ARIMA(1,1,1) converges toward ~5%.
