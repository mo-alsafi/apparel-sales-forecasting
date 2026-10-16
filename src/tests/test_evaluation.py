import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    MIN_TRAILING_OBS, STRESS_ORIGINS, TEST_ORIGINS, VALIDATION_ORIGINS,
    apply_scoring_mask, evaluate, get_forecast_folds, make_drift_forecaster,
    seasonal_naive_forecast, trailing_growth,
)

DATA_END = pd.Timestamp("2026-07-01")


def make_series():
    """Steadily growing monthly series spanning the project's date range."""
    index = pd.date_range("1992-01-01", periods=415, freq="MS")
    assert index[-1] == DATA_END
    return pd.Series(np.arange(415, dtype=float) + 100.0, index=index)


def target_months(origins):
    return {m for f in get_forecast_folds(origins) for m in f["forecast_months"]}


def test_forecast_window_is_aug_to_jul():
    fold = get_forecast_folds(["2016-07-31"])[0]
    assert fold["forecast_months"][0] == pd.Timestamp("2016-08-01")
    assert fold["forecast_months"][-1] == pd.Timestamp("2017-07-01")
    assert len(fold["forecast_months"]) == 12


def test_every_fold_starts_after_training_and_stays_inside_data():
    for fold in get_forecast_folds(VALIDATION_ORIGINS + STRESS_ORIGINS + TEST_ORIGINS):
        assert fold["forecast_months"].min() > fold["train_end"]
        assert fold["forecast_months"].max() <= DATA_END


def test_validation_stress_and_test_targets_are_disjoint():
    val, stress, test = (target_months(o) for o in (VALIDATION_ORIGINS, STRESS_ORIGINS, TEST_ORIGINS))
    assert not (val & stress) and not (val & test) and not (stress & test)


def test_test_period_is_the_last_24_months():
    months = sorted(target_months(TEST_ORIGINS))
    assert months[0] == pd.Timestamp("2024-08-01")
    assert months[-1] == DATA_END
    assert len(months) == 24


def test_scoring_mask_accepts_numpy_predictions():
    dates = pd.date_range("2020-02-01", periods=4, freq="MS")
    actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
    scored_y, scored_pred = apply_scoring_mask(actual, np.array([1.0, 2.0, 3.0, 4.0]))
    assert list(scored_y.index) == [pd.Timestamp("2020-02-01")]
    assert scored_y.index.equals(scored_pred.index)


def test_scoring_mask_can_be_disabled():
    dates = pd.date_range("2020-02-01", periods=4, freq="MS")
    actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
    scored_y, _ = apply_scoring_mask(actual, actual.values, mask=False)
    assert len(scored_y) == 4


def test_evaluate_passes_training_data_only_to_forecaster():
    y = pd.Series(np.arange(1.0, 121.0), index=pd.date_range("2010-01-01", periods=120, freq="MS"))
    seen = []

    def forecast_fn(train, forecast_months):
        seen.append((train.index.max(), forecast_months.copy()))
        return seasonal_naive_forecast(train, forecast_months)

    evaluate(forecast_fn, y, ["2016-07-31"])
    train_end, months = seen[0]
    assert train_end <= pd.Timestamp("2016-07-01")
    assert months[0] == pd.Timestamp("2016-08-01")
    assert months[-1] == pd.Timestamp("2017-07-01")


def test_test_origins_are_locked_unless_final():
    y = make_series()
    with pytest.raises(ValueError, match="locked"):
        evaluate(seasonal_naive_forecast, y, TEST_ORIGINS)
    assert len(evaluate(seasonal_naive_forecast, y, TEST_ORIGINS, final=True)) == 2


def test_trailing_drift_falls_back_to_zero_with_too_few_observations():
    y = make_series()
    train = y.loc[:"2022-07-01"]            # after masking, only one clean YoY row remains
    months = get_forecast_folds(["2022-07-31"])[0]["forecast_months"]
    assert trailing_growth(train)[1] < MIN_TRAILING_OBS
    drift = make_drift_forecaster("mean_trailing12")(train, months)
    assert np.allclose(drift.values, seasonal_naive_forecast(train, months).values)


def test_trailing_drift_applies_growth_when_enough_observations():
    y = make_series()
    train = y.loc[:"2018-07-01"]
    months = get_forecast_folds(["2018-07-31"])[0]["forecast_months"]
    assert trailing_growth(train)[1] >= MIN_TRAILING_OBS
    drift = make_drift_forecaster("mean_trailing12")(train, months)
    assert (drift.values > seasonal_naive_forecast(train, months).values).all()
