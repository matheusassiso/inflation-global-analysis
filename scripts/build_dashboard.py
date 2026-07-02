"""Generate the static interactive dashboard published at docs/index.html (GitHub Pages).

Reuses the same data pipeline as notebooks/01_inflation_global_analysis.ipynb.
Run: python scripts/build_dashboard.py
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA

RAW_PATH = "data/raw/Global Dataset of Inflation.csv"
OUT_PATH = "docs/index.html"
PREVIEW_PATH = "figures/07_dashboard_preview.png"

FOCUS = ["Brazil", "United States", "Germany", "Argentina", "Japan", "China"]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
REGIMES = [
    (1973, 1975, "Oil Shock", "red"),
    (1979, 1982, "Volcker", "orange"),
    (1990, 1995, "Post-Soviet Transition", "green"),
    (2007, 2009, "GFC", "gray"),
    (2021, 2022, "Post-COVID", "purple"),
]


def load_long():
    raw = pd.read_csv(RAW_PATH, encoding="latin1")
    wide = raw[raw["Series Name"] == "Headline Consumer Price Inflation"].copy()
    year_cols = [c for c in wide.columns if c.isdigit()]
    df = wide.melt(id_vars=["Country Code", "Country"], value_vars=year_cols,
                    var_name="year", value_name="inflation")
    df["year"] = df["year"].astype(int)
    df["inflation"] = pd.to_numeric(df["inflation"], errors="coerce")
    return df


def build():
    df = load_long()

    world = (df.groupby("year")["inflation"]
             .agg(median="median", mean="mean",
                  p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
             .reset_index())

    focus_df = df[df["Country"].isin(FOCUS)].copy()
    focus_df["inflation_plot"] = focus_df["inflation"].clip(upper=200)

    brazil = (focus_df[focus_df["Country"] == "Brazil"]
              .set_index("year")["inflation"].clip(upper=200).dropna())
    brazil_modern = brazil[brazil.index >= 1995]
    fit = ARIMA(brazil_modern, order=(1, 1, 1)).fit()
    fc = fit.get_forecast(steps=5)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=0.10)
    fc_years = list(range(brazil_modern.index[-1] + 1, brazil_modern.index[-1] + 6))

    fig = make_subplots(
        rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            "World Inflation — Median & IQR across 212 Countries (1970–2022)",
            "Selected Countries — Annual CPI Inflation (clipped at 200%)",
            "Brazil — ARIMA(1,1,1) Forecast (2023–2027)",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )

    # Row 1: world regimes
    fig.add_trace(go.Scatter(x=world["year"], y=world["p75"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=world["year"], y=world["p25"], fill="tonexty",
                              fillcolor="rgba(70,130,180,0.25)", line=dict(width=0),
                              name="IQR (25th–75th pct)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=world["year"], y=world["median"], line=dict(color="steelblue", width=2.5),
                              name="World Median"), row=1, col=1)
    fig.add_trace(go.Scatter(x=world["year"], y=world["mean"], line=dict(color="tomato", width=1.5, dash="dash"),
                              name="World Mean"), row=1, col=1)
    for start, end, label, color in REGIMES:
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.12, line_width=0,
                       annotation_text=label, annotation_position="top", row=1, col=1)

    # Row 2 col 1: selected countries
    for country, color in zip(FOCUS, COLORS):
        subset = focus_df[focus_df["Country"] == country]
        fig.add_trace(go.Scatter(x=subset["year"], y=subset["inflation_plot"],
                                  name=country, line=dict(color=color, width=1.6)),
                      row=2, col=1)

    # Row 2 col 2: ARIMA forecast
    fig.add_trace(go.Scatter(x=list(brazil_modern.index), y=list(brazil_modern.values),
                              name="Historical (1995–2022)", line=dict(color="#e41a1c", width=2)),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=fc_years + fc_years[::-1],
                              y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[:, 0])[::-1],
                              fill="toself", fillcolor="rgba(139,0,0,0.2)", line=dict(width=0),
                              mode="lines", name="90% CI", hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=fc_years, y=list(fc_mean.values), name="ARIMA Forecast",
                              line=dict(color="darkred", width=2, dash="dash"), mode="lines+markers"),
                  row=2, col=2)

    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=2)
    fig.update_layout(
        title="Global Inflation Dashboard (1970–2022) — Trends, Regimes, and Brazil ARIMA Forecast",
        height=850, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(t=90),
    )

    fig.write_html(OUT_PATH, include_plotlyjs="cdn", full_html=True)
    fig.write_image(PREVIEW_PATH, width=1280, height=850, scale=2)
    print(f"Wrote {OUT_PATH} and {PREVIEW_PATH}")


if __name__ == "__main__":
    build()
