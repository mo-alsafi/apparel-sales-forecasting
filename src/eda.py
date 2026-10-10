from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """Loads master CSV, sets DatetimeIndex at MS frequency, enforces 1992-01 to 2026-07 sales window,

    and asserts zero missing dates or index gaps.
    """
    df = pd.read_csv(csv_path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.set_index("observation_date").sort_index()

    # Enforce Month-Start frequency on the index
    df.index.freq = "MS"

    # Restrict core evaluation dataset to published sales window
    sales_series = df["apparel_sales_nsa"].dropna()
    start_date = pd.Timestamp("1992-01-01")
    end_date = pd.Timestamp("2026-07-01")

    assert (
        sales_series.index.min() == start_date
    ), f"Expected start date 1992-01-01, got {sales_series.index.min()}"
    assert (
        sales_series.index.max() == end_date
    ), f"Expected end date 2026-07-01, got {sales_series.index.max()}"

    expected_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    missing_dates = expected_range.difference(sales_series.index)
    assert (
        len(missing_dates) == 0
    ), f"Missing target observations detected: {missing_dates}"

    return df


def calculate_dec_jan_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the ratio of December sales in year t to January sales in year t+1.

    A stable ratio ~2.0x indicates structural multiplicative seasonality.
    """
    sales = df["apparel_sales_nsa"].dropna().to_frame()
    sales["year"] = sales.index.year
    sales["month"] = sales.index.month

    dec_sales = (
        sales[sales["month"] == 12]
        .set_index("year")["apparel_sales_nsa"]
        .rename("dec_sales")
    )
    jan_sales = (
        sales[sales["month"] == 1]
        .set_index("year")["apparel_sales_nsa"]
        .rename("jan_sales")
    )

    # Align December of year t with January of year t+1
    jan_next_year = jan_sales.shift(-1).rename("jan_next_year_sales")

    ratio_df = pd.concat([dec_sales, jan_next_year], axis=1).dropna()
    ratio_df["dec_jan_ratio"] = (
        ratio_df["dec_sales"] / ratio_df["jan_next_year_sales"]
    )

    return ratio_df


def run_stationarity_matrix(series: pd.Series, name: str) -> dict:
    """Executes ADF and KPSS unit root tests on a time series and returns a structured dictionary of statistics and p-values."""
    clean_series = series.dropna()

    # Augmented Dickey-Fuller Test (Null: Unit Root / Non-Stationary)
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(clean_series, autolag="AIC")

    # KPSS Test (Null: Stationarity around a constant)
    kpss_stat, kpss_p, _, kpss_crit = kpss(
        clean_series, regression="c", nlags="auto"
    )

    return {
        "Series Transform": name,
        "ADF Statistic": round(adf_stat, 4),
        "ADF p-value": round(adf_p, 4),
        "ADF Result (alpha=0.05)": (
            "Stationary" if adf_p < 0.05 else "Non-Stationary"
        ),
        "KPSS Statistic": round(kpss_stat, 4),
        "KPSS p-value": round(kpss_p, 4),
        "KPSS Result (alpha=0.05)": (
            "Non-Stationary" if kpss_p < 0.05 else "Stationary"
        ),
    }


def compute_exogenous_ccf(
    y_diff: pd.Series, x_diff: pd.Series, max_lag: int = 12
) -> pd.DataFrame:
    """Computes cross-correlation function (CCF) between target and exogenous regressor at lags 0 through max_lag on differenced series to prevent spurious correlation."""
    aligned = pd.concat([y_diff, x_diff], axis=1).dropna()
    y_vals = aligned.iloc[:, 0].values
    x_vals = aligned.iloc[:, 1].values

    ccf_values = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(y_vals, x_vals)[0, 1]
        else:
            corr = np.corrcoef(y_vals[lag:], x_vals[:-lag])[0, 1]
        ccf_values.append({"lag": lag, "cross_correlation": round(corr, 4)})

    return pd.DataFrame(ccf_values)