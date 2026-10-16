import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    VALIDATION_ORIGINS, TEST_ORIGINS, apply_scoring_mask, evaluate,
    get_forecast_folds, seasonal_naive_forecast,
)


def test_forecast_window_is_aug_to_jul():
    fold = get_forecast_folds(["2016-07-31"])[0]
    assert fold["forecast_months"][0] == pd.Timestamp("2016-08-01")
    assert fold["forecast_months"][-1] == pd.Timestamp("2017-07-01")


def test_folds_stay_inside_data_and_after_training():
    for fold in get_forecast_folds(VALIDATION_ORIGINS + TEST_ORIGINS):
        assert fold["forecast_months"].min() > fold["train_end"]
        assert fold["forecast_months"].max() <= pd.Timestamp("2026-07-01")


def test_scoring_mask_accepts_numpy_predictions():
    dates = pd.date_range("2020-02-01", periods=4, freq="MS")
    actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)
    scored_y, scored_pred = apply_scoring_mask(actual, np.array([1.0, 2.0, 3.0, 4.0]))
    assert list(scored_y.index) == [pd.Timestamp("2020-02-01")]
    assert scored_y.index.equals(scored_pred.index)


def test_evaluate_passes_training_data_only_to_forecaster():
    y = pd.Series(
        np.arange(1.0, 121.0),
        index=pd.date_range("2010-01-01", periods=120, freq="MS"),
    )
    seen = []

    def forecast_fn(train, forecast_months):
        seen.append((train.index.max(), forecast_months.copy()))
        return seasonal_naive_forecast(train, forecast_months)

    evaluate(forecast_fn, y, ["2016-07-31"])
    train_end, months = seen[0]
    assert train_end <= pd.Timestamp("2016-07-01")
    assert months[0] == pd.Timestamp("2016-08-01")
    assert months[-1] == pd.Timestamp("2017-07-01")
