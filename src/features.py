
"""Leakage-safe macroeconomic features for apparel sales forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_SET_COLUMNS = {
    "unemployment": ["unrate_change12_pp"],
    "cpi": ["cpi_yoy_pct"],
    "both": ["unrate_change12_pp", "cpi_yoy_pct"],
}


def _month_start_index(values) -> pd.DatetimeIndex:
    """Normalize timestamps to monthly period starts."""
    index = pd.DatetimeIndex(pd.to_datetime(values))

    if index.hasnans:
        raise ValueError("Date index contains invalid timestamps.")

    return index.to_period("M").to_timestamp()


def _prepare_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the complete macroeconomic dataset.

    The input must contain the columns 'cpi' and 'unrate'.
    Earlier observations must be retained so that the first
    sales observations can receive valid lagged features.
    """
    df = macro.copy()

    if "observation_date" in df.columns:
        df["observation_date"] = pd.to_datetime(
            df["observation_date"]
        )
        df = df.set_index("observation_date")

    df.index = _month_start_index(df.index)
    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError("Duplicate monthly macro observations found.")

    required = {"cpi", "unrate"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required macro columns: {sorted(missing)}"
        )

    # Ensure the macro dataset has a continuous monthly index.
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS",
    )
    df = df.reindex(full_index)

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def transform_macro_features(
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the agreed transformations on the full macro series.

    unrate_change12_pp:
        Current unemployment rate minus its value 12 months earlier.
        The result is measured in percentage points.

    cpi_yoy_pct:
        CPI percentage growth versus the same month one year earlier.
        The result is expressed as a percentage.

    No rows are dropped or masked before these transformations.
    """
    df = _prepare_macro(macro)

    transformed = pd.DataFrame(index=df.index)

    transformed["unrate_change12_pp"] = (
        df["unrate"].diff(periods=12)
    )

    transformed["cpi_yoy_pct"] = (
        df["cpi"]
        .pct_change(periods=12, fill_method=None)
        * 100.0
    )

    return transformed


def build_regressor_matrices(
    macro: pd.DataFrame,
    train_dates,
    forecast_dates,
    origin,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build training and forecasting exogenous matrices.

    For a target month t, both features are taken from t - 12 months.

    All macro transformations are calculated before the 12-month
    shift. Forecast regressors must be fully observable on or before
    the forecast origin.

    Parameters
    ----------
    macro:
        Complete monthly macro dataset, including pre-1992 history.
    train_dates:
        Index of the training target series.
    forecast_dates:
        Exact forecast-month index.
    origin:
        Last month available at forecast time.
    feature_set:
        One of 'unemployment', 'cpi', or 'both'.

    Returns
    -------
    X_train, X_forecast:
        DataFrames with matching columns and correctly aligned dates.
    """
    if feature_set not in FEATURE_SET_COLUMNS:
        raise ValueError(
            "feature_set must be 'unemployment', 'cpi', or 'both'."
        )

    train_index = _month_start_index(train_dates)
    forecast_index = _month_start_index(forecast_dates)
    origin_month = _month_start_index([origin])[0]

    if train_index.empty or forecast_index.empty:
        raise ValueError("Training and forecast dates cannot be empty.")

    if train_index.has_duplicates or forecast_index.has_duplicates:
        raise ValueError("Duplicate target months are not allowed.")

    if not train_index.is_monotonic_increasing:
        raise ValueError("Training dates must be sorted.")

    if not forecast_index.is_monotonic_increasing:
        raise ValueError("Forecast dates must be sorted.")

    if (train_index > origin_month).any():
        raise ValueError("Training data extends beyond the origin.")

    if (forecast_index <= origin_month).any():
        raise ValueError("Every forecast month must follow the origin.")

    # Exact 12-month lag: at target date t, the macro transformation
    # must be dated t - 12 months or earlier.
    source_dates = forecast_index - pd.DateOffset(months=12)

    if (source_dates > origin_month).any():
        raise ValueError(
            "A forecast regressor would require macro data "
            "from after the forecast origin."
        )

    # Transform the full series first, then shift it by 12 rows.
    # _prepare_macro guarantees a continuous monthly index, so this
    # row shift is exactly a 12-calendar-month shift.
    transformed = transform_macro_features(macro)
    lagged_features = transformed.shift(periods=12)

    columns = FEATURE_SET_COLUMNS[feature_set]

    X_train = lagged_features.reindex(train_index)[columns].copy()
    X_forecast = (
        lagged_features.reindex(forecast_index)[columns].copy()
    )

    # Refuse to silently fit a model with incomplete regressors.
    for label, matrix in (
        ("training", X_train),
        ("forecast", X_forecast),
    ):
        values = matrix.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            bad_dates = matrix.index[
                ~np.isfinite(values).all(axis=1)
            ]

            raise ValueError(
                f"Non-finite {label} regressor values for "
                f"{feature_set}: "
                f"{bad_dates.strftime('%Y-%m').tolist()[:8]}"
            )

    return X_train, X_forecast
