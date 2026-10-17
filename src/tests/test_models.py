import numpy as np
import pandas as pd
import pytest

from src.models import (
    ets_forecast,
    sarima_raw_forecast,
    sarima_log_forecast,
    sarima_month_dummy_log_forecast,
    make_prophet_forecaster,
    _ets_training_series,
    COVID_MONTHS,
)


def make_training_series():
    """Synthetic monthly sales with trend, seasonality, and a shock."""
    index = pd.date_range(
        "1992-01-01",
        periods=372,
        freq="MS",
    )

    t = np.arange(len(index))

    # January trough and December peak.
    monthly_factors = np.array([
        0.65, 0.72, 0.90, 0.93,
        0.98, 0.94, 0.91, 1.03,
        0.96, 1.00, 1.20, 1.80,
    ])

    values = (
        (9000.0 + 18.0 * t)
        * monthly_factors[index.month - 1]
    )

    # Artificial shock. The model functions must not fit these
    # four actual values directly.
    shock_values = {
        "2020-03-01": 10600.0,
        "2020-04-01": 2600.0,
        "2020-05-01": 8400.0,
        "2020-06-01": 15700.0,
    }

    for date, value in shock_values.items():
        values[index == pd.Timestamp(date)] = value

    return pd.Series(
        values,
        index=index,
        name="apparel_sales_nsa",
    )


@pytest.fixture
def training_data():
    return make_training_series()


MODEL_CASES = [
    ("ETS", ets_forecast),
    ("SARIMA raw", sarima_raw_forecast),
    ("SARIMA log", sarima_log_forecast),
    (
        "SARIMA month dummies",
        sarima_month_dummy_log_forecast,
    ),
    (
        "Prophet additive",
        make_prophet_forecaster("additive", 0.05),
    ),
    (
        "Prophet multiplicative",
        make_prophet_forecaster("multiplicative", 0.05),
    ),
]


@pytest.mark.parametrize(
    "model_name, forecast_fn",
    MODEL_CASES,
)
def test_model_forecast_contract(
    model_name,
    forecast_fn,
    training_data,
):
    """Every model returns exactly 12 finite, positive forecasts."""

    train = training_data.copy()

    forecast_months = pd.date_range(
        train.index.max() + pd.offsets.MonthBegin(1),
        periods=12,
        freq="MS",
    )

    prediction = forecast_fn(train, forecast_months)

    assert isinstance(prediction, pd.Series), model_name
    assert prediction.index.equals(forecast_months), model_name
    assert len(prediction) == 12
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction > 0).all()


def test_ets_covid_imputation_is_finite_and_does_not_mutate_input(
    training_data,
):
    """ETS fills the four COVID months using the year-ago values."""

    original = training_data.copy()
    filled = _ets_training_series(training_data)

    assert training_data.equals(original)
    assert filled.loc[COVID_MONTHS].notna().all()
    assert np.isfinite(filled.loc[COVID_MONTHS]).all()
    assert (filled.loc[COVID_MONTHS] > 0).all()

    # The synthetic shock values should have been replaced.
    assert not np.allclose(
        filled.loc[COVID_MONTHS].to_numpy(),
        original.loc[COVID_MONTHS].to_numpy(),
    )