
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

@dataclass(frozen=True)
class StrategicChampionResult:
    summary: pd.DataFrame
    daily: pd.DataFrame
    selection: pd.DataFrame

LAGS: Tuple[int, ...] = (1, 7, 14, 28, 56)
ROLLING_WINDOWS: Tuple[int, ...] = (7, 14, 28, 56)

def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)

def _aligned_arrays(actual, predicted):
    a = pd.to_numeric(
        pd.Series(actual).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    p = pd.to_numeric(
        pd.Series(predicted).reset_index(drop=True),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    if len(a) != len(p):
        raise ValueError(
            f"Metric length mismatch: actual={len(a)}, predicted={len(p)}"
        )

    return a, p

def _wape(actual, predicted):
    a, p = _aligned_arrays(actual, predicted)
    den = float(np.abs(a).sum())
    return 0.0 if den == 0 else float(np.abs(p-a).sum()/den*100.0)

def _bias_pct(actual, predicted):
    a, p = _aligned_arrays(actual, predicted)
    den = float(a.sum())
    return 0.0 if den == 0 else float((p.sum()-a.sum())/den*100.0)

def _portfolio_series(historical, cutoff, metric):
    f = historical.copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    f[metric] = _num(f[metric])
    f = f.loc[f["date"] <= cutoff].copy()
    s = f.groupby("date")[metric].sum().sort_index()
    idx = pd.date_range(s.index.min(), cutoff, freq="D")
    return s.reindex(idx, fill_value=0.0).astype(float)

def _calendar_features(dates):
    df = pd.DataFrame(index=dates)
    df["dow"] = dates.dayofweek.astype(int)
    df["month"] = dates.month.astype(int)
    df["dayofyear"] = dates.dayofyear.astype(int)
    df["weekofyear"] = dates.isocalendar().week.astype(int).to_numpy()
    df["trend"] = np.arange(len(dates), dtype=float)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7.0)
    df["doy_sin"] = np.sin(2*np.pi*df["dayofyear"]/365.25)
    df["doy_cos"] = np.cos(2*np.pi*df["dayofyear"]/365.25)
    return df

def _supervised_frame(series):
    idx = pd.DatetimeIndex(series.index)
    df = _calendar_features(idx)
    df["target"] = series.to_numpy(dtype=float)
    for lag in LAGS:
        df[f"lag_{lag}"] = series.shift(lag).to_numpy()
    shifted = series.shift(1)
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean().to_numpy()
        df[f"roll_median_{window}"] = shifted.rolling(window, min_periods=window).median().to_numpy()
    return df.dropna().copy()

def _feature_row(history, forecast_date, trend_value):
    v = history.astype(float)
    d = pd.Timestamp(forecast_date)
    row = {
        "dow": int(d.dayofweek), "month": int(d.month), "dayofyear": int(d.dayofyear),
        "weekofyear": int(d.isocalendar().week), "trend": float(trend_value),
        "dow_sin": float(np.sin(2*np.pi*d.dayofweek/7.0)),
        "dow_cos": float(np.cos(2*np.pi*d.dayofweek/7.0)),
        "doy_sin": float(np.sin(2*np.pi*d.dayofyear/365.25)),
        "doy_cos": float(np.cos(2*np.pi*d.dayofyear/365.25)),
    }
    for lag in LAGS:
        row[f"lag_{lag}"] = float(v.iloc[-lag])
    for window in ROLLING_WINDOWS:
        w = v.iloc[-window:]
        row[f"roll_mean_{window}"] = float(w.mean())
        row[f"roll_median_{window}"] = float(w.median())
    return pd.DataFrame([row])

def _forecast_hgbr(train_series, future_dates):
    sf = _supervised_frame(train_series)
    if len(sf) < 90:
        raise ValueError(f"Not enough supervised rows: {len(sf)}")
    X, y = sf.drop(columns=["target"]), sf["target"]
    model = HistGradientBoostingRegressor(
        learning_rate=0.04, max_iter=300, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42
    )
    model.fit(X, y)
    hist = train_series.copy()
    preds = []
    base = float(len(train_series))
    for step, date in enumerate(future_dates):
        row = _feature_row(hist, pd.Timestamp(date), base + step)[X.columns]
        pred = max(0.0, float(model.predict(row)[0]))
        preds.append(pred)
        hist.loc[pd.Timestamp(date)] = pred
    return np.asarray(preds, dtype=float)

def _forecast_weekday_median(train_series, future_dates, weeks=8):
    recent = train_series.tail(min(len(train_series), weeks*7))
    overall = float(recent.median())
    levels = {
        dow: (float(recent.loc[recent.index.dayofweek == dow].median())
              if len(recent.loc[recent.index.dayofweek == dow]) else overall)
        for dow in range(7)
    }
    if len(train_series) >= 56:
        prev = float(train_series.iloc[-56:-28].median())
        curr = float(train_series.iloc[-28:].median())
        ratio = curr/prev if prev > 0 else 1.0
    else:
        ratio = 1.0
    ratio = float(np.clip(ratio, 0.90, 1.10))
    growth = np.log(ratio)/28.0
    phi = 0.96
    out = []
    for step, date in enumerate(future_dates, start=1):
        damped = (1.0-phi**step)/(1.0-phi)
        factor = float(np.exp(growth*damped))
        out.append(max(0.0, levels[int(pd.Timestamp(date).dayofweek)]*factor))
    return np.asarray(out, dtype=float)

def _forecast_weekly_repeat(train_series, future_dates):
    recent = train_series.tail(min(28, len(train_series)))
    levels = {}
    for dow in range(7):
        vals = recent.loc[recent.index.dayofweek == dow]
        levels[dow] = float(vals.mean()) if len(vals) else float(recent.mean())
    return np.asarray(
        [max(0.0, levels[int(pd.Timestamp(d).dayofweek)]) for d in future_dates],
        dtype=float
    )

def _candidate_forecasts(train_series, future_dates):
    return {
        "PortfolioHGBR": _forecast_hgbr(train_series, future_dates),
        "WeekdayMedian8W": _forecast_weekday_median(train_series, future_dates, weeks=8),
        "WeeklyRepeat4W": _forecast_weekly_repeat(train_series, future_dates),
    }

def _select_candidate(full_pre_cutoff_series, horizon_days):
    inner_days = min(60, max(28, int(horizon_days)//2))
    if len(full_pre_cutoff_series) < 56 + inner_days + 30:
        raise ValueError("Not enough pre-cutoff history for inner validation.")
    train = full_pre_cutoff_series.iloc[:-inner_days].copy()
    actual = full_pre_cutoff_series.iloc[-inner_days:].copy()
    dates = pd.DatetimeIndex(actual.index)
    candidates = _candidate_forecasts(train, dates)
    rows = []
    for name, pred in candidates.items():
        ps = pd.Series(pred, index=dates)
        rows.append({
            "Candidate": name,
            "InnerValidationDays": inner_days,
            "WAPE": round(_wape(actual, ps), 4),
            "BiasPct": round(_bias_pct(actual, ps), 4),
        })
    ranking = pd.DataFrame(rows).sort_values(["WAPE","Candidate"]).reset_index(drop=True)
    return str(ranking.iloc[0]["Candidate"]), ranking

def evaluate_strategic_champion(historical, baseline_daily, cutoff_date, horizon_days):
    horizon = int(horizon_days)
    baseline = baseline_daily.copy()
    baseline["ForecastDate"] = pd.to_datetime(baseline["ForecastDate"], errors="coerce").dt.normalize()
    baseline = baseline.sort_values("ForecastDate").head(horizon).reset_index(drop=True)
    dates = pd.DatetimeIndex(baseline["ForecastDate"])
    out = baseline.copy()
    selections, summaries = [], []

    for metric, pred_col, actual_col in (
        ("clicks","PredictedClicks","ActualClicks"),
        ("impressions","PredictedImpressions","ActualImpressions"),
    ):
        pre = _portfolio_series(historical, cutoff_date, metric)
        champion, ranking = _select_candidate(pre, horizon)
        ranking.insert(0, "Metric", metric)
        ranking.insert(0, "HorizonDays", horizon)
        ranking["Selected"] = ranking["Candidate"].eq(champion)
        selections.append(ranking)

        final_candidates = _candidate_forecasts(pre, dates)
        champion_pred = pd.Series(final_candidates[champion], index=dates)
        out[f"Champion{metric.title()}"] = champion_pred.to_numpy()

        actual = _num(out[actual_col])
        base_pred = _num(out[pred_col])

        summaries.extend([
            {
                "HorizonDays": horizon, "Metric": metric, "Method": "RecursiveDailyML",
                "SelectedCandidate": "", "TotalErrorPct": round(abs(_bias_pct(actual, base_pred)),2),
                "WAPE": round(_wape(actual, base_pred),2),
                "BiasPct": round(_bias_pct(actual, base_pred),2),
            },
            {
                "HorizonDays": horizon, "Metric": metric, "Method": "StrategicChampionPortfolio",
                "SelectedCandidate": champion, "TotalErrorPct": round(abs(_bias_pct(actual, champion_pred)),2),
                "WAPE": round(_wape(actual, champion_pred),2),
                "BiasPct": round(_bias_pct(actual, champion_pred),2),
            }
        ])

    out["ChampionImpressions"] = np.maximum(_num(out["ChampionImpressions"]), _num(out["ChampionClicks"]))
    return StrategicChampionResult(
        summary=pd.DataFrame(summaries),
        daily=out,
        selection=pd.concat(selections, ignore_index=True),
    )
