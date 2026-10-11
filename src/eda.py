import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ---------------------------------------------------------------------------
# Analysis windows (the COVID decision is documented in the README)
# ---------------------------------------------------------------------------
PRE_COVID_END = "2019-12-01"                  # last month used to identify model orders
COVID_ACUTE = ("2020-03-01", "2020-06-01")    # months masked as missing when fitting models
COVID_EXCLUDE = ("2020-03-01", "2021-12-01")  # shock + 12-month base effect, excluded from correlations


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

    A ratio that stays constant would mean the seasonal swing scales with the sales level.
    In this data it is NOT constant: it falls from ~2.5x in the 1990s to ~2.1x after 2007,
    i.e. the seasonal pattern is flattening (see summarize_dec_jan_ratio).
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

    # statsmodels emits FutureWarning (return-type change) and InterpolationWarning
    # (KPSS p-values are clipped to the lookup table range 0.01 - 0.10) on every call.
    # Both are known and harmless here, so keep the notebook output readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Augmented Dickey-Fuller Test (Null: Unit Root / Non-Stationary)
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(clean_series, autolag="AIC")

        # KPSS Test (Null: Stationarity around a constant)
        # NOTE: reported p-values are bounded by the table: 0.01 means "<= 0.01", 0.10 means ">= 0.10".
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
    y_diff: pd.Series,
    x_diff: pd.Series,
    max_lag: int = 12,
    exclude: tuple | None = None,
) -> pd.DataFrame:
    """Cross-correlation between a stationary target and a stationary exogenous series.

    Lag k means corr(y_t, x_{t-k}): the regressor LEADS the target by k months.
    Both inputs must already be differenced / growth-rate series (never raw levels).

    exclude: optional (start, end) date strings. Those months are blanked IN PLACE (not dropped),
    so the remaining pairs stay correctly aligned in time after the lag shift.
    """
    aligned = pd.concat([y_diff, x_diff], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    if exclude is not None:
        aligned.loc[exclude[0]:exclude[1]] = np.nan

    y_vals = aligned["y"].to_numpy(dtype=float)
    x_vals = aligned["x"].to_numpy(dtype=float)

    rows = []
    for lag in range(max_lag + 1):
        yy, xx = (y_vals, x_vals) if lag == 0 else (y_vals[lag:], x_vals[:-lag])
        ok = ~(np.isnan(yy) | np.isnan(xx))
        corr = float(np.corrcoef(yy[ok], xx[ok])[0, 1])
        rows.append({"lag": lag, "cross_correlation": round(corr, 4), "n_obs": int(ok.sum())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers added in the Phase 2 revision (pure pandas / numpy unless noted)
# ---------------------------------------------------------------------------
def acf_pacf_table(series: pd.Series, lags=(1, 2, 3, 4, 10, 11, 12, 13, 14, 24), nlags: int = 36):
    """ACF / PACF values at selected lags plus the approximate 95% band (+/- 1.96 / sqrt(n)).

    Uses statsmodels' acf (biased estimator) and pacf(method='ywm'), the same estimators
    that plot_acf / plot_pacf draw, so the table matches the figure.
    """
    x = series.dropna().to_numpy(dtype=float)
    a = acf(x, nlags=nlags)
    p = pacf(x, nlags=nlags, method="ywm")
    table = pd.DataFrame(
        {"lag": list(lags), "acf": [round(float(a[k]), 3) for k in lags], "pacf": [round(float(p[k]), 3) for k in lags]}
    )
    band = round(1.96 / np.sqrt(len(x)), 3)
    return table, band


def summarize_dec_jan_ratio(
    ratio_df: pd.DataFrame, periods=((1992, 1999), (2000, 2006), (2007, 2019), (2020, 2025))
):
    """Mean December/January ratio per period and the linear trend (ratio points per year)."""
    r = ratio_df["dec_jan_ratio"]
    table = pd.DataFrame(
        [{"period": f"{a}-{b}", "mean_ratio": round(float(r.loc[a:b].mean()), 2)} for a, b in periods]
    )
    slope = float(np.polyfit(r.index.to_numpy(dtype=float), r.to_numpy(dtype=float), 1)[0])
    return table, slope


def seasonal_index(sales: pd.Series, end: str = PRE_COVID_END) -> pd.Series:
    """Each month's sales as a share of its calendar-year mean (1.0 = an average month).

    Dividing by the annual mean removes the long-run growth, so a boxplot of this index by
    month shows seasonality only. Restricted to complete pre-COVID years.
    """
    s = sales.loc[:end]
    return s / s.groupby(s.index.year).transform("mean")


def seasonal_amplitude_diagnostic(sales: pd.Series, exclude_years=(2020, 2021)):
    """Does the seasonal swing grow in proportion to the sales level?

    Fits  log(sd of monthly sales within a year) = a + b * log(mean level of that year).
      b ~ 1  -> swing proportional to level (multiplicative -> log transform is right)
      b ~ 0  -> swing independent of level (additive -> no transform)
    Rule of thumb: the variance-stabilising Box-Cox lambda is 1 - b.
    Only complete calendar years are used; exclude_years drops the COVID years.
    """
    by_year = sales.groupby(sales.index.year)
    per_year = pd.DataFrame(
        {
            "mean_level": by_year.mean(),
            "sd_level": by_year.std(),
            "sd_log": np.log(sales).groupby(sales.index.year).std(),
            "n_months": by_year.size(),
        }
    )
    per_year = per_year[per_year["n_months"] == 12].drop(index=list(exclude_years), errors="ignore")
    slope = float(np.polyfit(np.log(per_year["mean_level"]), np.log(per_year["sd_level"]), 1)[0])
    return per_year.drop(columns="n_months"), slope, 1.0 - slope


def growth_vs_base_year(sales: pd.Series, year: int, base_year: int = 2019) -> pd.Series:
    """Same-month growth (%) of `year` versus `base_year`, indexed by month number 1-12.

    Using a normal pre-COVID base year avoids the 2021 'base effect' of comparing with 2020.
    """
    cur = sales[sales.index.year == year].copy()
    base = sales[sales.index.year == base_year].copy()
    cur.index, base.index = cur.index.month, base.index.month
    return ((cur / base) - 1.0) * 100.0


def period_growth(sales: pd.Series, cpi: pd.Series, year: int, base_year: int) -> dict:
    """Annual nominal sales growth, all-items CPI inflation, and CPI-deflated (real) growth, in %.

    Compares calendar-year totals of sales with calendar-year mean CPI. Both years must be complete.
    """
    for y in (year, base_year):
        if int((sales.index.year == y).sum()) != 12:
            raise ValueError(f"{y} is not a complete year in the sales series")
    ann = sales.groupby(sales.index.year).sum()
    cpi_a = cpi.groupby(cpi.index.year).mean()
    nominal = ann[year] / ann[base_year] - 1.0
    inflation = cpi_a[year] / cpi_a[base_year] - 1.0
    real = (ann[year] / cpi_a[year]) / (ann[base_year] / cpi_a[base_year]) - 1.0
    return {
        "comparison": f"{year} vs {base_year}",
        "nominal_sales_%": round(100 * float(nominal), 1),
        "cpi_inflation_%": round(100 * float(inflation), 1),
        "real_sales_%": round(100 * float(real), 1),
    }


def covid_impact(
    sales: pd.Series,
    shock_month: str = "2020-04-01",
    ref_month: str = "2020-02-01",
    norm_years=(2017, 2018, 2019),
) -> dict:
    """Three ways to size the COVID shock (all in %): naive, seasonally aware, and year-over-year.

    naive_ref_to_shock: shock month vs the last pre-shock month (ignores seasonality).
    vs_seasonal_norm  : shock month vs what the usual shock/ref seasonal ratio would imply.
    yoy               : shock month vs the same month one year earlier.
    """
    shock, ref = pd.Timestamp(shock_month), pd.Timestamp(ref_month)
    ratios = [
        sales[pd.Timestamp(year=y, month=shock.month, day=1)] / sales[pd.Timestamp(year=y, month=ref.month, day=1)]
        for y in norm_years
    ]
    expected = sales[ref] * float(np.mean(ratios))
    return {
        "naive_ref_to_shock_%": round(100 * float(sales[shock] / sales[ref] - 1), 2),
        "vs_seasonal_norm_%": round(100 * float(sales[shock] / expected - 1), 1),
        "yoy_%": round(100 * float(sales[shock] / sales[shock - pd.DateOffset(years=1)] - 1), 1),
        "seasonal_ratio": round(float(np.mean(ratios)), 2),
        "expected_without_covid": round(float(expected)),
    }
