
import pandas as pd
from src.evaluation import get_forecast_folds


def make_mock_data():
    """Create a monthly timeline matching the project span."""
    dates = pd.date_range(
        start="1992-01-01",
        end="2026-07-01",
        freq="MS",
    )
    return pd.DataFrame(
        {"apparel_sales_nsa": 1000.0},
        index=dates,
    )


def test_no_future_training_leakage():
    """Training observations must be strictly before each forecast origin."""
    df_mock = make_mock_data()

    origins = [
        "2016-07-31",
        "2017-07-31",
        "2018-07-31",
        "2019-07-31",
        "2022-07-31",
        "2023-07-31",
    ]

    folds = get_forecast_folds(origins)

    assert folds, "No forecast folds were generated."

    for fold in folds:
        origin = fold["origin"]
        train_data = df_mock.loc[
            df_mock.index <= fold["train_end"]
        ]

        assert not train_data.empty, (
            f"Training set for {origin} is empty."
        )

        assert train_data.index.max() < origin, (
            f"Temporal leakage: training data ends at "
            f"{train_data.index.max()}, origin is {origin}."
        )


def test_validation_and_test_targets_do_not_overlap():
    """Validation target months must not overlap test target months."""
    val_origins = [
        "2016-07-31",
        "2017-07-31",
        "2018-07-31",
        "2019-07-31",
    ]

    test_origins = [
        "2022-07-31",
        "2023-07-31",
    ]

    val_folds = get_forecast_folds(val_origins)
    test_folds = get_forecast_folds(test_origins)

    val_months = {
        pd.Timestamp(month)
        for fold in val_folds
        for month in fold["forecast_months"]
    }

    test_months = {
        pd.Timestamp(month)
        for fold in test_folds
        for month in fold["forecast_months"]
    }

    overlap = val_months.intersection(test_months)

    assert not overlap, (
        "Validation/test target overlap detected: "
        f"{sorted(overlap)}"
    )


def test_imputed_october_2025_excluded_from_earlier_training():
    """An October 2025 imputation must not enter earlier training windows."""
    df_mock = make_mock_data()
    imputed_date = pd.Timestamp("2025-10-01")

    origins = [
        "2024-07-31",
        "2025-07-31",
    ]

    folds = get_forecast_folds(origins)

    for fold in folds:
        origin = fold["origin"]
        train_data = df_mock.loc[
            df_mock.index <= fold["train_end"]
        ]

        if origin < imputed_date:
            assert imputed_date not in train_data.index, (
                f"October 2025 appears in training data "
                f"for origin {origin}."
            )
