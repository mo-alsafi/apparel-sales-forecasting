
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import VALIDATION_ORIGINS
from src.features import (
    build_regressor_matrices,
    transform_macro_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MACRO_PATH = REPO_ROOT / "reports" / "00-monthly_macro_apparel.csv"


def load_macro_for_test():
    df = pd.read_csv(
        MACRO_PATH,
        parse_dates=["observation_date"],
    )
    return df.set_index("observation_date").sort_index()


def make_fold_dates(origin):
    """Build training dates and the next 12 August-July months."""
    origin = (
        pd.Timestamp(origin)
        .to_period("M")
        .to_timestamp()
    )

    train_dates = pd.date_range(
        "1992-01-01",
        origin,
        freq="MS",
    )

    forecast_dates = pd.date_range(
        start=origin + pd.offsets.MonthBegin(1),
        periods=12,
        freq="MS",
    )

    return origin, train_dates, forecast_dates


def test_post_origin_macro_changes_do_not_change_inputs():
    """Changing future macro observations cannot alter fold inputs."""
    macro = load_macro_for_test()

    origin, train_dates, forecast_dates = make_fold_dates(
        "2019-07-31"
    )

    original_train, original_forecast = build_regressor_matrices(
        macro,
        train_dates,
        forecast_dates,
        origin,
        "both",
    )

    changed = macro.copy()
    future_rows = changed.index > origin

    # Dramatically change both relevant macro series after the origin.
    # No training or forecast regressor should change.
    changed.loc[future_rows, "cpi"] = (
        changed.loc[future_rows, "cpi"] + 1000.0
    )

    changed.loc[future_rows, "unrate"] = (
        changed.loc[future_rows, "unrate"] + 25.0
    )

    changed_train, changed_forecast = build_regressor_matrices(
        changed,
        train_dates,
        forecast_dates,
        origin,
        "both",
    )

    pd.testing.assert_frame_equal(
        original_train,
        changed_train,
    )

    pd.testing.assert_frame_equal(
        original_forecast,
        changed_forecast,
    )


def test_forecast_regressors_use_exactly_12_months_of_lag():
    """Each forecast row must equal the transformation from t - 12."""
    macro = load_macro_for_test()

    origin, train_dates, forecast_dates = make_fold_dates(
        "2019-07-31"
    )

    transformed = transform_macro_features(macro)

    _, X_forecast = build_regressor_matrices(
        macro,
        train_dates,
        forecast_dates,
        origin,
        "both",
    )

    for target_date in forecast_dates:
        source_date = (
            target_date - pd.DateOffset(months=12)
        )

        for column in X_forecast.columns:
            expected = transformed.loc[source_date, column]
            actual = X_forecast.loc[target_date, column]

            assert np.isclose(actual, expected), (
                f"{column} at {target_date:%Y-%m} must use "
                f"{source_date:%Y-%m}."
            )


def test_october_2025_imputations_never_enter_validation_rows():
    """
    The imputed October 2025 CPI and unemployment values cannot
    enter any Phase 5 validation input row.
    """
    macro = load_macro_for_test()

    imputed_month = pd.Timestamp("2025-10-01")

    # Confirm this guard corresponds to the current repository snapshot.
    row = macro.loc[imputed_month]

    assert int(row["cpi_is_imputed"]) == 1
    assert int(row["unrate_is_imputed"]) == 1

    for origin_text in VALIDATION_ORIGINS:
        origin, train_dates, forecast_dates = make_fold_dates(
            origin_text
        )

        target_rows = train_dates.append(forecast_dates)

        # Unemployment change and CPI YoY growth each depend on
        # macro values at the transformation date and 12 months
        # before that transformation date.
        dependencies_12 = (
            target_rows - pd.DateOffset(months=12)
        )
        dependencies_24 = (
            target_rows - pd.DateOffset(months=24)
        )

        assert not (dependencies_12 == imputed_month).any()
        assert not (dependencies_24 == imputed_month).any()

        # Also build both matrices to verify that their values
        # are present and finite for every validation fold.
        X_train, X_forecast = build_regressor_matrices(
            macro,
            train_dates,
            forecast_dates,
            origin,
            "both",
        )

        assert np.isfinite(X_train.to_numpy()).all()
        assert np.isfinite(X_forecast.to_numpy()).all()
