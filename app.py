"""Streamlit app: explore GP appointment forecasts by NHS region and model.

Run locally:  streamlit run app.py
Deploy free:  share.streamlit.io -> New app -> this repo -> app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

OUT = Path(__file__).parent / "outputs"

st.set_page_config(page_title="GP Appointment Demand Forecast", layout="wide")


@st.cache_data
def load():
    hist = pd.read_csv(OUT / "history_monthly.csv", parse_dates=["month"])
    fc = pd.read_csv(OUT / "forecast_monthly.csv", parse_dates=["month"])
    scores = pd.read_csv(OUT / "backtest_summary.csv")
    return hist, fc, scores


hist, fc, scores = load()

st.title("GP appointment demand forecast, England")
st.caption("NHS England *Appointments in General Practice* daily data, Mar 2024 – Aug 2026. "
           "Forecasts Sep 2026 – Feb 2027, chosen by rolling back-test.")

regions = ["England"] + sorted(r for r in hist.region.unique() if r != "England")
models = scores.groupby("model").mape_monthly.mean().sort_values().index.tolist()
c1, c2 = st.columns([1, 1])
region = c1.selectbox("Region", regions)
model = c2.selectbox("Model", models, help="Ordered by back-test accuracy; the first is the best.")

h = hist[hist.region == region]
f = fc[(fc.region == region) & (fc.model == model)]
err = scores[(scores.region == region) & (scores.model == model)].mape_monthly.iloc[0]
last_year = h[(h.month >= "2025-09-01") & (h.month <= "2026-02-01")].actual.sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Forecast, Sep 2026 – Feb 2027", f"{f.yhat.sum() / 1e6:.1f}M")
k2.metric("Change vs same months last year", f"{100 * (f.yhat.sum() / last_year - 1):+.1f}%")
peak = f.loc[f.yhat.idxmax()]
k3.metric("Peak month", f"{peak.month:%b %Y}", f"{peak.yhat / 1e6:.1f}M appointments", delta_color="off")
k4.metric("Back-test error (monthly MAPE)", f"{err:.2f}%")

actual = alt.Chart(h).mark_line(point=True, color="#2a78d6").encode(
    x=alt.X("month:T", title=None),
    y=alt.Y("actual:Q", title="Appointments", scale=alt.Scale(zero=False)),
    tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("actual:Q", format=",.0f", title="Actual")])
band = alt.Chart(f).mark_area(opacity=0.3, color="#86b6ef").encode(
    x="month:T", y="lower:Q", y2="upper:Q")
line = alt.Chart(f).mark_line(point=alt.OverlayMarkDef(color="#eb6834"), strokeDash=[5, 3], color="#eb6834").encode(
    x="month:T", y="yhat:Q",
    tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("yhat:Q", format=",.0f", title="Forecast"),
             alt.Tooltip("lower:Q", format=",.0f", title="80% low"), alt.Tooltip("upper:Q", format=",.0f", title="80% high")])
st.altair_chart((band + actual + line).properties(height=380), width="stretch")
st.caption("Blue: actual monthly appointments. Orange dashed: forecast. Shaded: 80% range based on the model's own back-test errors.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Forecast table")
    t = f[["month", "yhat", "lower", "upper"]].copy()
    t["month"] = t.month.dt.strftime("%b %Y")
    st.dataframe(t.rename(columns={"month": "Month", "yhat": "Forecast", "lower": "80% low", "upper": "80% high"})
                 .style.format({"Forecast": "{:,.0f}", "80% low": "{:,.0f}", "80% high": "{:,.0f}"}),
                 hide_index=True, width="stretch")
with right:
    st.subheader(f"Model accuracy: {region}")
    s = scores[scores.region == region][["model", "mape_monthly", "mape_weekday"]].sort_values("mape_monthly")
    st.dataframe(s.rename(columns={"model": "Model", "mape_monthly": "Monthly MAPE %",
                                   "mape_weekday": "Weekday daily MAPE %"}),
                 hide_index=True, width="stretch")
    st.caption("4 rolling back-tests, each forecasting 3 months ahead (Sep 2025 – Aug 2026).")
