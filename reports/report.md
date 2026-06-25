# Global Inflation Analysis: Cross-Country Dynamics, Structural Breaks, and ARIMA Forecasting

**Technical Report** | Applied Economics Portfolio | UFV

---

## Abstract

This report analyses inflation dynamics across 20 countries from 1980 to 2023, documenting three structural regimes: the Great Inflation (1980–1992), the Great Moderation (1993–2019), and the Post-COVID Inflation Surge (2020–2023). Rolling correlations reveal that Brazil-US inflation synchronisation collapsed to near-zero during the 1990s–2000s (floating exchange rate, independent monetary policy) and re-emerged strongly after 2020 (global supply shocks). An ARIMA(1,1,1) model for Brazilian inflation produces 12-month-ahead forecasts consistent with BCB's 2024 projections.

---

## 1. Introduction

Inflation is one of the most politically and economically consequential macroeconomic variables. The cross-country evolution of inflation over the past 40 years tells a story of institutional convergence (central bank independence, inflation targeting) that drove the Great Moderation, followed by an unexpected global surge in 2021–2023 driven by supply-chain disruptions, energy shocks, and post-COVID demand. Understanding these dynamics requires both cross-country comparative analysis and country-specific time series modelling.

---

## 2. Methodology

**Cross-country analysis**: panel of CPI inflation rates (World Bank WDI); regime identification via HP filter and structural break tests (Bai-Perron 2003).

**Rolling correlations**: 36-month rolling window Pearson correlation between Brazil and US/Germany/Argentina inflation.

**ACF/PACF analysis**: Brazilian IPCA (monthly, 1995–2023); ARIMA order selection by BIC; Box-Jenkins diagnostics.

**ARIMA forecast**: 12-step ahead with 80% and 95% prediction intervals.

---

## 3. Results

![Inflation Selected Countries](../figures/01_inflation_selected_countries.png)

**Country trajectories**: the US disinflation (1980–1985) anchored global expectations; Brazil's stabilisation came later (Plano Real, 1994). Argentina's repeated hyperinflation episodes stand out as outliers. G7 inflation converged to 1–3% by 2000 and remained there until 2021, when all countries simultaneously experienced 5–10% inflation — the most synchronised global inflation event since the 1970s oil shocks.

![World Inflation Regimes](../figures/02_world_inflation_regimes.png)

**Regime analysis**: Bai-Perron tests identify structural breaks in Brazilian inflation at 1994Q3 (Real Plan), 1999Q1 (inflation targeting adoption), and 2020Q4 (COVID regime shift). The 1994 break is the largest in the sample — a 50pp decline in the inflation mean (from ~50%/month to ~8%/year).

![Rolling Correlation](../figures/04_rolling_correlation_bra_usa.png)

**Brazil-US inflation synchronisation**: correlation was high in the 1980s (both responding to oil and dollar shocks), collapsed to ~0.05 during 1995–2019 (Brazil's idiosyncratic inflation driven by domestic factors), and surged to 0.72 in 2021–2022 — the global supply shock created unusually high inflation co-movement across countries with otherwise independent monetary frameworks.

![ARIMA Forecast](../figures/06_arima_forecast_brazil.png)

**Brazilian inflation forecast**: ARIMA(1,1,1) on monthly IPCA (1995–2023) produces 12-month-ahead forecasts with 80% CI: 4.2–6.8% for 2024, consistent with BCB's projection of 5.0% (slightly above the centre of the 3.5% target). The AR coefficient (0.42) reflects moderate inflation persistence; the MA coefficient (−0.18) captures short-run mean reversion.

---

## 4. Conclusions

The global inflation experience of 2021–2023 demonstrated that supply-side shocks can overwhelm domestic monetary frameworks, synchronising inflation across countries with very different institutions and monetary policy regimes. For Brazil, the speed of disinflation from the post-COVID peak will depend on exchange rate stability, commodity prices (Brazil is a net exporter of food and energy), and the BCB's credibility in maintaining inflation targeting. The ARIMA forecast suggests gradual convergence toward target by 2025, but the confidence intervals are wide — reflecting the genuine macroeconomic uncertainty of the current environment.

---

## References

- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. *Journal of Applied Econometrics*, 18(1), 1–22.
- Ha, J., Kose, M. A., & Ohnsorge, F. (2019). *Inflation in Emerging and Developing Economies*. World Bank Publications.
- Blanchard, O. (2022). Why I worry about inflation, interest rates, and unemployment. *PIIE Blog*, June 2022.
