# Global Inflation: Trends, Regimes, and Cross-Country Dynamics (1970–2022)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4e9a06)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

> **Research question:** How did global inflation evolve across five decades, and do countries inflate together?

Part of the [Applied Economics Research Portfolio](https://github.com/matheusassiso) — Project 01 of 55 | Section: EDA & Descriptive Analysis

---

## Overview

This project applies the full data science workflow to the IMF/World Bank Global Inflation Dataset (212 countries, 1970–2022). Starting from a wide-format panel, it reshapes the data, characterises inflation by decade and regime, maps cross-country synchronisation, and culminates in an ARIMA(1,1,1) forecast for Brazilian inflation through 2027.

| | |
|---|---|
| **Dataset** | Global Inflation Dataset — 212 countries × 52 years (IMF/World Bank via Kaggle) |
| **Methods** | EDA · Descriptive statistics · Time series · Correlation · ADF · ARIMA |
| **Libraries** | `pandas` `numpy` `matplotlib` `seaborn` `statsmodels` |

---

## Key Findings

- Global inflation peaked in the **1970s–80s** (oil shocks); the world median exceeded 10% from 1973–82.
- The **1990s disinflation** was synchronised across advanced and emerging economies — driven by central bank independence and inflation targeting.
- The **2021–22 post-COVID surge** was the first globally synchronised inflation spike since the 1970s, with Brazil-US rolling correlation jumping to 0.72.
- Brazil's post-Real Plan regime (post-1995) is mean-reverting; **ARIMA(1,1,1)** converges toward ~5% by 2027 with wide CIs reflecting genuine macro uncertainty.

---

## Figures

### 1 — Inflation trajectories for 6 countries (1970–2022)
![Inflation Selected Countries](figures/01_inflation_selected_countries.png)

---

### 2 — Global median ± IQR with annotated inflation regimes
![World Inflation Regimes](figures/02_world_inflation_regimes.png)

---

### 3 — Pearson correlation matrix across selected economies
![Correlation Heatmap](figures/03_correlation_heatmap.png)

---

### 4 — 10-year rolling correlation: Brazil vs USA
![Rolling Correlation](figures/04_rolling_correlation_bra_usa.png)

---

### 5 — ACF / PACF diagnostics for ARIMA order selection
![ACF PACF](figures/05_acf_pacf_brazil.png)

---

### 6 — ARIMA(1,1,1) forecast for Brazil (2023–2027)
![ARIMA Forecast](figures/06_arima_forecast_brazil.png)

---

## Methods

| Step | Description |
|---|---|
| Data reshape | Wide-to-long panel (212 countries × 52 years) |
| Descriptive stats | Mean, median, std, skewness by decade |
| Regime visualisation | Time series with annotated oil shocks, Volcker, GFC, COVID |
| Global envelope | World median ± IQR across all countries |
| Cross-country correlation | Pearson matrix for 6 selected countries |
| Rolling correlation | 10-year window — Brazil vs USA (1980–2022) |
| Stationarity | ADF test on Brazilian inflation series |
| Time series model | ARIMA(1,1,1) — BIC selection, ACF/PACF diagnostics |
| Forecast | 5-year horizon with 90% confidence interval |

---

## Data

**Source:** [Global Inflation Dataset — Kaggle](https://www.kaggle.com/datasets/belayethossainds/global-inflation-dataset-212-country-19702022)  
212 countries · 1970–2022 · IMF/World Bank

```bash
# Requires a Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d belayethossainds/global-inflation-dataset-212-country-19702022 \
  --path data/raw --unzip
```

Raw data is excluded from version control (`.gitignore`). Place the CSV in `data/raw/` before running.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_inflation_global_analysis.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_inflation_global_analysis.ipynb
```

---

## Repository Structure

```
inflation-global-analysis/
├── data/
│   └── raw/               # place CSV here (not tracked)
├── figures/               # all 6 output figures (PNG)
├── notebooks/
│   └── 01_inflation_global_analysis.ipynb
├── reports/
│   └── report.md          # full technical write-up
├── requirements.txt
└── README.md
```

---

## Stack

`Python 3.12` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `statsmodels`

---

*Part of a 55-project Applied Economics portfolio built during the MSc in Applied Economics at UFV (Universidade Federal de Viçosa).*
