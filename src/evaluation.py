"""Leakage-safe monthly forecast evaluation utilities."""
from __future__ import annotations
from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd

VALIDATION_ORIGINS = (
    "2016-07-31",
    "2017-07-31",
    "2018-07-31",
    "2019-07-31",
    "2022-07-31",
    "2023-07-31",
)

STRESS_ORIGINS = (
    "2020-07-31",
    "2021-07-31",
)

TEST_ORIGINS = (
    "2024-07-31",
    "2025-07-31",
)

ALL_ORIGINS = (
    VALIDATION_ORIGINS
    + STRESS_ORIGINS
    + TEST_ORIGINS
)

MASK_START = pd.Timestamp("2020-03-01")
MASK_END = pd.Timestamp("2021-06-30")
MAX_FORECAST_MONTH = pd.Timestamp("2026-07-01")

# Minimum number of clean recent YoY observations for drift.
MIN_TRAILING_OBS = 6


def is_masked(index):
    idx = pd.DatetimeIndex(index)
    return pd.Series((idx >= MASK_START) & (idx <= MASK_END), index=idx)


def get_forecast_folds(origins: Iterable[str | pd.Timestamp] = ALL_ORIGINS, *,
                       data_end: str | pd.Timestamp = MAX_FORECAST_MONTH) -> list[dict]:
    """Create 12-month August-to-July folds from July origins."""
    data_end = pd.Timestamp(data_end).to_period("M").to_timestamp()
    folds = []
    for origin in origins:
        origin_ts = pd.Timestamp(origin)
        train_end = origin_ts.to_period("M").to_timestamp()
        start = origin_ts + pd.offsets.MonthBegin(1)
        forecast_months = pd.date_range(start=start, periods=12, freq="MS")
        if forecast_months.min() <= train_end:
            raise ValueError(f"Forecast must begin after training end: {origin_ts}")
        if forecast_months.max() > data_end:
            raise ValueError(f"Fold from {origin_ts:%Y-%m-%d} ends after data end {data_end:%Y-%m-%d}")
        folds.append({"origin": origin_ts, "train_end": train_end,
                      "forecast_months": forecast_months})
    return folds


def apply_scoring_mask(
    y_true: pd.Series,
    y_pred,
    *,
    mask: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Align actuals and forecasts and apply the primary scoring mask."""

    actual = pd.Series(y_true).copy()
    actual.index = pd.DatetimeIndex(actual.index)

    if isinstance(y_pred, pd.Series):
        pred = y_pred.reindex(actual.index)
    else:
        values = np.asarray(y_pred, dtype=float)

        if len(values) != len(actual):
            raise ValueError(
                "Predictions and actuals must have equal lengths."
            )

        pred = pd.Series(values, index=actual.index)

    keep = actual.notna() & pred.notna()

    if mask:
        keep &= ~is_masked(actual.index).to_numpy()

    return actual.loc[keep], pred.loc[keep]


def trailing_growth(train: pd.Series) -> tuple[float, int]:
    """Mean eligible YoY growth from the trailing 12 calendar months."""

    train = pd.Series(train).sort_index()

    if train.empty:
        return 0.0, 0

    growth = eligible_yoy_growth(train)
    start = train.index.max() - pd.DateOffset(months=11)
    trailing = growth.loc[growth.index >= start].dropna()

    if trailing.empty:
        return 0.0, 0

    return float(trailing.mean()), int(len(trailing))


def calculate_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def calculate_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


def calculate_bias(y_true, y_pred) -> float:
    """Mean forecast minus actual; negative means under-forecasting."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def calculate_rel_mae(y_true, y_pred, y_benchmark) -> float:
    denominator = calculate_mae(y_true, y_benchmark)
    return float("nan") if denominator == 0 else calculate_mae(y_true, y_pred) / denominator


def seasonal_naive_forecast(train: pd.Series, forecast_months: pd.DatetimeIndex) -> pd.Series:
    """Repeat the training observation from 12 months before each forecast month."""
    train = pd.Series(train).sort_index()
    source_months = forecast_months - pd.DateOffset(years=1)
    return train.reindex(source_months).set_axis(forecast_months)


def eligible_yoy_growth(train: pd.Series) -> pd.Series:
    """Exclude YoY growth rows whose current or lag-12 date is masked."""
    train = pd.Series(train).sort_index()
    yoy = train.pct_change(12)
    current_masked = is_masked(yoy.index).to_numpy()
    prior_masked = is_masked(yoy.index - pd.DateOffset(years=1)).to_numpy()
    return yoy.loc[~(current_masked | prior_masked)].dropna()


def make_drift_forecaster(method: str) -> Callable:
    """Build seasonal-naive-plus-drift forecast: mean_all, median_all, mean_trailing12."""
    if method not in {"mean_all", "median_all", "mean_trailing12"}:
        raise ValueError("method must be mean_all, median_all, or mean_trailing12")
    
    fallback_origins = set()

    def forecast(train: pd.Series, forecast_months: pd.DatetimeIndex) -> pd.Series:
        sn = seasonal_naive_forecast(train, forecast_months)
        growth = eligible_yoy_growth(train)
        if method == "mean_all":
            rate = float(growth.mean()) if len(growth) else 0.0
        elif method == "median_all":
            rate = float(growth.median()) if len(growth) else 0.0
        else:   
            start = train.index.max() - pd.DateOffset(months=11)
            trailing = growth.loc[growth.index >= start]

            if len(trailing) >= MIN_TRAILING_OBS:
                rate = float(trailing.mean())
            else:
                # Insufficient evidence: do not extrapolate a noisy drift estimate.
                rate = 0.0
                forecast.fallback_origins.add(train.index.max())

        return sn * (1 + rate)

    forecast.fallback_origins = fallback_origins
    
    forecast.__name__ = f"seasonal_naive_plus_{method}"
    return forecast


def evaluate(
    forecast_fn: Callable,
    y: pd.Series,
    origins: Iterable[str | pd.Timestamp] = VALIDATION_ORIGINS,
    *,
    final: bool = False,
    mask: bool = True,
    predictions_out: list | None = None,
) -> pd.DataFrame:
    """
    Evaluate forecasts on expanding 12-month August-to-July folds.

    Test origins are rejected unless final=True is explicit.
    predictions_out, when supplied, receives each monthly forecast,
    actual value, error, and scoring-mask status.
    """

    origins = list(origins)

    requested_months = {
        pd.Timestamp(origin).to_period("M")
        for origin in origins
    }
    locked_test_months = {
        pd.Timestamp(origin).to_period("M")
        for origin in TEST_ORIGINS
    }

    if not final and requested_months & locked_test_months:
        raise ValueError(
            "Test origins are locked. Use final=True only for "
            "the final evaluation after model selection."
        )

    y = pd.Series(y).copy()
    y.index = pd.DatetimeIndex(y.index)
    y = y.sort_index()

    folds = get_forecast_folds(
        origins,
        data_end=y.index.max(),
    )

    rows = []

    for fold in folds:
        months = fold["forecast_months"]

        # The forecast function receives data only through the origin.
        train = y.loc[:fold["train_end"]].copy()
        actual = y.reindex(months)

        raw_pred = forecast_fn(train, months)

        # Enforce the forecast-function contract.
        if isinstance(raw_pred, pd.Series):
            if not raw_pred.index.equals(months):
                raise ValueError(
                    f"{forecast_fn.__name__} returned the wrong "
                    "forecast index."
                )
            pred = raw_pred.astype(float)
        else:
            values = np.asarray(raw_pred, dtype=float)

            if len(values) != len(months):
                raise ValueError(
                    f"{forecast_fn.__name__} returned the wrong "
                    "number of predictions."
                )

            pred = pd.Series(values, index=months)

        benchmark = seasonal_naive_forecast(train, months)

        scored_actual, scored_pred = apply_scoring_mask(
            actual, pred, mask=mask
        )
        scored_benchmark_actual, scored_benchmark = (
            apply_scoring_mask(actual, benchmark, mask=mask)
        )

        if not scored_actual.index.equals(
            scored_benchmark_actual.index
        ):
            raise ValueError(
                "Model and seasonal-naive scoring months differ."
            )

        # Retain forecasts for later diagnostics, including masked months.
        if predictions_out is not None:
            for date in months:
                actual_value = actual.loc[date]
                forecast_value = pred.loc[date]

                date_is_masked = bool(
                    is_masked(pd.DatetimeIndex([date])).iloc[0]
                )

                finite_pair = (
                    pd.notna(actual_value)
                    and pd.notna(forecast_value)
                    and np.isfinite(actual_value)
                    and np.isfinite(forecast_value)
                )

                scored = (
                    finite_pair
                    and (not mask or not date_is_masked)
                )

                error = (
                    float(forecast_value - actual_value)
                    if finite_pair
                    else np.nan
                )

                predictions_out.append({
                    "Model": getattr(
                        forecast_fn, "__name__", "forecast"
                    ),
                    "Origin": fold["origin"].strftime("%Y-%m"),
                    "Date": date,
                    "Actual": actual_value,
                    "Forecast": forecast_value,
                    "Error": error,
                    "AbsError": (
                        abs(error) if finite_pair else np.nan
                    ),
                    "Scored": bool(scored),
                    "Mask_Excluded": bool(
                        mask and date_is_masked
                    ),
                })

        december_months = months[months.month == 12]

        dec_actual, dec_pred = apply_scoring_mask(
            actual.reindex(december_months),
            pred.reindex(december_months),
            mask=mask,
        )

        if len(dec_actual) and (dec_actual != 0).all():
            december_pct_error = float(
                (
                    (dec_pred - dec_actual).abs()
                    / dec_actual.abs()
                ).mean()
                * 100
            )
        else:
            december_pct_error = float("nan")

        rows.append({
            "Origin": fold["origin"].strftime("%Y-%m"),
            "Train_End": fold["train_end"].strftime("%Y-%m"),
            "Scored_Months": len(scored_actual),
            "MAE": calculate_mae(
                scored_actual, scored_pred
            ),
            "Benchmark_SN_MAE": calculate_mae(
                scored_actual, scored_benchmark
            ),
            "RMSE": calculate_rmse(
                scored_actual, scored_pred
            ),
            "Bias": calculate_bias(
                scored_actual, scored_pred
            ),
            "RelMAE_vs_SN": calculate_rel_mae(
                scored_actual,
                scored_pred,
                scored_benchmark,
            ),
            "December_Abs_Pct_Error": december_pct_error,
        })

    return pd.DataFrame(rows)