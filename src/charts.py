"""README charts: back-test accuracy by model, and the England forecast."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, DOCS = ROOT / "outputs", ROOT / "docs"
DOCS.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
BLUE, ORANGE, BAND = "#2a78d6", "#eb6834", "#cde2fb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.axisbelow": True,
})

# 1. Back-test accuracy
s = pd.read_csv(OUT / "backtest_summary.csv").groupby("model").mape_monthly.mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 3.2))
colors = [BLUE if m == s.idxmin() else "#9ec5f4" for m in s.index]
ax.barh(s.index, s.values, color=colors, height=0.55)
for i, v in enumerate(s.values):
    ax.text(v + 0.08, i, f"{v:.2f}%", va="center", fontsize=9.5, color=INK2)
ax.set_xlabel("Mean absolute % error on monthly totals (4 back-tests × 8 series)")
ax.tick_params(axis="y", colors=INK2)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_title("The simplest model wins: last year's pattern × recent growth", loc="left", fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig(DOCS / "model_accuracy.png", dpi=160)
plt.close(fig)

# 2. England history + forecast
h = pd.read_csv(OUT / "history_monthly.csv", parse_dates=["month"])
f = pd.read_csv(OUT / "forecast_monthly.csv", parse_dates=["month"])
h = h[h.region == "England"]
f = f[(f.region == "England") & (f.model == "Seasonal naive + growth")]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(h.month, h.actual / 1e6, color=BLUE, lw=2, marker="o", ms=3.5, label="Actual")
bridge = pd.concat([h.tail(1).rename(columns={"actual": "yhat"}), f])
ax.fill_between(f.month, f.lower / 1e6, f.upper / 1e6, color=BAND, label="80% range (from back-test errors)")
ax.plot(bridge.month, bridge.yhat / 1e6, color=ORANGE, lw=2, ls=(0, (4, 2)), marker="o", ms=3.5, label="Forecast")
peak = f.loc[f.yhat.idxmax()]
ax.annotate(f"Oct 2026: {peak.yhat/1e6:.1f}M forecast", (peak.month, peak.yhat / 1e6), xytext=(-170, 6),
            textcoords="offset points", color=INK2, fontsize=9)
ax.set_ylabel("Appointments (millions)")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_ylim(20, None)
ax.legend(frameon=False, loc="lower left", ncols=3)
ax.set_title("GP appointments in England: actual Mar 2024 – Aug 2026, forecast to Feb 2027", loc="left",
             fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig(DOCS / "england_forecast.png", dpi=160)
plt.close(fig)
print("charts saved")
