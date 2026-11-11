"""Back-test four forecasting models on daily GP appointments, then forecast 6 months ahead.

Models
* Seasonal naive          : same weekday 52 weeks earlier (the baseline to beat)
* Seasonal naive + growth : last year's value x recent year-on-year growth (trailing 13 weeks)
* SARIMA                  : SARIMAX(1,0,1)(1,0,0,7) on the log ratio to last year's same weekday,
                            with bank-holiday and Christmas flags (this year and last) as regressors
* Prophet                 : weekly + yearly seasonality (Fourier order 15), multiplicative,
                            England bank holidays and Christmas period

Evaluation: rolling-origin back-test, 4 origins x 92-day horizon (3 complete months each), scored with MAPE on
monthly totals (what capacity planners use) and on weekday daily counts.
"""
from pathlib import Path
import json
import logging
import warnings

import holidays
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "processed", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ORIGINS = ["2025-08-31", "2025-11-30", "2026-02-28", "2026-05-31"]
HORIZON = 92
FUTURE_END = "2027-02-28"

ENG_HOLIDAYS = holidays.UK(subdiv="ENG", years=range(2024, 2028))


def load_series() -> dict[str, pd.Series]:
    d = pd.read_parquet(DATA / "daily_region.parquet")
    series = {r: g.set_index("date")["appointments"].asfreq("D", fill_value=0) for r, g in d.groupby("region")}
    series["England"] = d.groupby("date")["appointments"].sum().asfreq("D", fill_value=0)
    return series


def is_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([d in ENG_HOLIDAYS for d in index.date])


def is_christmas(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([(d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 2) for d in index])


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Bank holiday and Christmas flags for each day and for the same weekday last year."""
    ly = index - pd.Timedelta(days=364)
    return pd.DataFrame({"bank_holiday": is_holiday(index).astype(int),
                         "christmas": is_christmas(index).astype(int),
                         "bank_holiday_last_year": is_holiday(ly).astype(int),
                         "christmas_last_year": is_christmas(ly).astype(int)}, index=index)


def seasonal_naive(train: pd.Series, future: pd.DatetimeIndex) -> pd.DataFrame:
    vals = [train.get(d - pd.Timedelta(days=364), np.nan) for d in future]
    return pd.DataFrame({"yhat": vals}, index=future)


def seasonal_naive_growth(train: pd.Series, future: pd.DatetimeIndex) -> pd.DataFrame:
    recent = train[-91:]
    year_ago = train.reindex(recent.index - pd.Timedelta(days=364))
    growth = recent.sum() / year_ago.sum()
    base = seasonal_naive(train, future)
    return base * growth


def sarima(train: pd.Series, future: pd.DatetimeIndex) -> pd.DataFrame:
    """SARIMA on the log ratio to the same weekday last year.

    Dividing by last year removes the yearly pattern (October vaccination peak, August dip);
    the model then learns how this year's ratio moves, with calendar effects as regressors.
    """
    idx = train.index[train.index >= train.index.min() + pd.Timedelta(days=364)]
    last_year = train.reindex(idx - pd.Timedelta(days=364)).values
    ratio = pd.Series(np.log1p(train[idx].values) - np.log1p(last_year), index=idx)
    fit = SARIMAX(ratio, exog=calendar(idx), order=(1, 0, 1), seasonal_order=(1, 0, 0, 7),
                  trend="c").fit(disp=False, maxiter=300)
    fc = fit.get_forecast(len(future), exog=calendar(future))
    base = np.log1p(train.reindex(future - pd.Timedelta(days=364)).values)
    ci = fc.conf_int(alpha=0.2)
    return pd.DataFrame({"yhat": np.expm1(base + fc.predicted_mean.values),
                         "lower": np.expm1(base + ci.iloc[:, 0].values),
                         "upper": np.expm1(base + ci.iloc[:, 1].values)}, index=future)


def prophet(train: pd.Series, future: pd.DatetimeIndex) -> pd.DataFrame:
    hol = pd.DataFrame({"holiday": "bank_holiday", "ds": pd.to_datetime(list(ENG_HOLIDAYS.keys()))})
    xmas = pd.DataFrame({"holiday": "christmas_period",
                         "ds": pd.to_datetime([f"{y}-12-24" for y in range(2024, 2028)]),
                         "lower_window": 0, "upper_window": 9})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=15, daily_seasonality=False,
                seasonality_mode="multiplicative", holidays=pd.concat([hol, xmas]), interval_width=0.8)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    p = m.predict(pd.DataFrame({"ds": future}))
    return pd.DataFrame({"yhat": p.yhat.clip(lower=0).values, "lower": p.yhat_lower.clip(lower=0).values,
                         "upper": p.yhat_upper.values}, index=future)


MODELS = {"Seasonal naive": seasonal_naive, "Seasonal naive + growth": seasonal_naive_growth,
          "SARIMA": sarima, "Prophet": prophet}


def mape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mask = actual > 0
    return float(100 * np.mean(np.abs(actual[mask] - pred[mask]) / actual[mask]))


def backtest(series: dict) -> pd.DataFrame:
    rows, preds = [], []
    for region, y in series.items():
        for origin in ORIGINS:
            o = pd.Timestamp(origin)
            train = y[:o]
            future = pd.date_range(o + pd.Timedelta(days=1), periods=HORIZON, freq="D")
            actual = y.reindex(future)
            for name, fn in MODELS.items():
                f = fn(train, future)
                weekday = (future.dayofweek < 5) & ~is_holiday(future) & ~is_christmas(future)
                monthly = pd.DataFrame({"a": actual, "p": f.yhat}).resample("MS").sum()
                complete = monthly.index[(monthly.index >= future[0]) &
                                         (monthly.index + pd.offsets.MonthEnd(0) <= future[-1])]
                monthly = monthly.loc[complete]
                rows.append({"region": region, "origin": origin, "model": name,
                             "mape_monthly": mape(monthly.a, monthly.p),
                             "mape_weekday": mape(actual[weekday], f.yhat[weekday])})
                preds.append(pd.DataFrame({"date": future, "region": region, "origin": origin, "model": name,
                                           "actual": actual.values, "forecast": f.yhat.values}))
        print(f"back-tested {region}")
    pd.concat(preds).to_parquet(OUT / "backtest_predictions.parquet", index=False)
    return pd.DataFrame(rows)


def final_forecast(series: dict, scores: pd.DataFrame) -> pd.DataFrame:
    """Monthly forecasts Sep 2026 - Feb 2027 from every model.

    The 80% interval comes from back-test errors: the 80th percentile of absolute monthly
    percentage error for that region and model, so it reflects how wrong the model has been.
    """
    q80 = scores.groupby(["region", "model"]).mape_monthly.quantile(0.8) / 100
    out = []
    for region, y in series.items():
        future = pd.date_range(y.index.max() + pd.Timedelta(days=1), FUTURE_END, freq="D")
        for name, fn in MODELS.items():
            m = fn(y, future)[["yhat"]].resample("MS").sum()
            err = q80[(region, name)]
            m["lower"], m["upper"] = m.yhat * (1 - err), m.yhat * (1 + err)
            m["region"], m["model"] = region, name
            out.append(m.reset_index(names="month"))
    return pd.concat(out)


def main():
    series = load_series()
    bt = backtest(series)
    bt.to_csv(OUT / "backtest_scores.csv", index=False)
    summary = bt.groupby(["region", "model"])[["mape_monthly", "mape_weekday"]].mean().round(2).reset_index()
    summary.to_csv(OUT / "backtest_summary.csv", index=False)
    print(summary.pivot(index="region", columns="model", values="mape_monthly").to_string())
    print(summary.pivot(index="region", columns="model", values="mape_weekday").to_string())

    best = summary.loc[summary.groupby("region").mape_monthly.idxmin()].set_index("region").model.to_dict()
    fc = final_forecast(series, bt)
    fc.to_csv(OUT / "forecast_monthly.csv", index=False)
    history = pd.concat([s.resample("MS").sum().rename("actual").to_frame().assign(region=r)
                         for r, s in series.items()]).reset_index(names="month")
    history.to_csv(OUT / "history_monthly.csv", index=False)

    overall = bt.groupby("model")[["mape_monthly", "mape_weekday"]].mean().round(2)
    (OUT / "metrics.json").write_text(json.dumps({
        "data_from": str(series["England"].index.min().date()),
        "data_to": str(series["England"].index.max().date()),
        "origins": ORIGINS, "horizon_days": HORIZON,
        "overall_mape": overall.to_dict(orient="index"),
        "best_model_by_region": best,
    }, indent=2))
    print(overall.to_string())
    print(best)


if __name__ == "__main__":
    main()
