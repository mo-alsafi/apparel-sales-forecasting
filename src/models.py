"""Training-only forecasting functions for Phase 4."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.features import build_regressor_matrices


logger = logging.getLogger(__name__)

COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2020-06-01")

COVID_MONTHS = pd.date_range(
    COVID_START,
    COVID_END,
    freq="MS",
)

SEASONAL_SARIMA_ORDERS = (
    (0, 1, 1),
    (0, 1, 2),
)

MONTH_DUMMY_ORDERS = (
    (0, 1, 1),
    (0, 1, 2),
)

SEASONAL_ORDER = (0, 1, 1, 12)
NO_SEASONAL_DIFFERENCING = (0, 0, 0, 0)


def _monthly_train(train: pd.Series) -> pd.Series:
    """Return a sorted, continuous month-start series."""

    series = pd.Series(train, dtype=float).copy()

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    if series.empty:
        raise ValueError("Training data is empty.")

    series.index = series.index.to_period("M").to_timestamp()
    series = series.sort_index()

    if series.index.has_duplicates:
        raise ValueError("Training data has duplicate months.")

    full_index = pd.date_range(
        series.index.min(),
        series.index.max(),
        freq="MS",
    )

    return series.reindex(full_index).astype(float)


def _forecast_series(
    values,
    forecast_months: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    """Enforce the forecast contract and positive dollar outputs."""

    months = pd.DatetimeIndex(forecast_months)
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(values) != len(months):
        raise ValueError(
            f"{name} produced {len(values)} values for "
            f"{len(months)} forecast months."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} produced a non-finite forecast."
        )

    # All sales forecasts must be positive.
    # Clip only non-positive point forecasts to a tiny positive floor.
    if (values <= 0).any():
        raise ValueError(
            f"{name} produced non-positive forecasts. "
            "Inspect the fitted model and its transformation."
        )

    return pd.Series(values, index=months, name=name)


def _mask_covid(train: pd.Series) -> pd.Series:
    """
    Replace March-June 2020 by NaN for models supporting missing targets.

    This is a training transformation. It is not the scoring mask.
    """
    series = _monthly_train(train)
    series.loc[COVID_MONTHS.intersection(series.index)] = np.nan
    return series


def _ets_training_series(train: pd.Series) -> pd.Series:
    """
    ETS cannot accept the four missing COVID observations.

    Impute each available March-June 2020 observation as its value
    from the same month in 2019 multiplied by (1 + trailing growth).

    Trailing growth is the mean YoY growth during March 2019-February
    2020, estimated only from the supplied training data.
    """
    original = _monthly_train(train)
    series = original.copy()

    months_present = COVID_MONTHS.intersection(series.index)

    if len(months_present) == 0:
        return series

    yoy = original.pct_change(12)

    growth_window = yoy.loc[
        (yoy.index >= pd.Timestamp("2019-03-01"))
        & (yoy.index <= pd.Timestamp("2020-02-01"))
    ].dropna()

    trailing_growth = (
        float(growth_window.mean())
        if len(growth_window)
        else 0.0
    )

    growth_multiplier = max(0.01, 1.0 + trailing_growth)

    for month in months_present:
        previous_year = month - pd.DateOffset(years=1)

        if previous_year not in original.index:
            raise ValueError(
                f"Cannot impute {month:%Y-%m}: "
                "the year-ago month is not available."
            )

        reference_value = original.loc[previous_year]

        if (
            not np.isfinite(reference_value)
            or reference_value <= 0
        ):
            raise ValueError(
                f"Invalid year-ago value for {month:%Y-%m}."
            )

        series.loc[month] = (
            reference_value * growth_multiplier
        )

    if series.isna().any():
        raise ValueError(
            "ETS received unexpected missing training values. "
            "Only March-June 2020 are imputed automatically."
        )

    return series


def ets_forecast(
    train: pd.Series,
    forecast_months: pd.DatetimeIndex,
) -> pd.Series:
    """
    Holt-Winters ETS forecast.

    Fit four candidate specifications on training data only:
      * additive seasonality, ordinary trend
      * additive seasonality, damped trend
      * multiplicative seasonality, ordinary trend
      * multiplicative seasonality, damped trend

    Select the lowest training AICc.
    """
    series = _ets_training_series(train)
    months = pd.DatetimeIndex(forecast_months)

    candidates = []

    specifications = [
        ("add", False),
        ("add", True),
        ("mul", False),
        ("mul", True),
    ]

    for seasonal, damped in specifications:
        label = f"ETS seasonal={seasonal}, damped={damped}"

        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=damped,
                seasonal=seasonal,
                seasonal_periods=12,
                initialization_method="estimated",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = model.fit(optimized=True)

            for warning in caught:
                logger.warning(
                    "%s warning (%s): %s",
                    label,
                    warning.category.__name__,
                    warning.message,
                )

            score = float(fitted.aicc)

            if np.isfinite(score):
                candidates.append((score, label, fitted))
            else:
                logger.warning(
                    "%s returned a non-finite AICc.", label
                )

        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)

    if not candidates:
        raise RuntimeError(
            "All ETS specifications failed. "
            "Review the warnings above."
        )

    _, selected_label, selected_fit = min(
        candidates, key=lambda item: item[0]
    )

    logger.info(
        "Selected %s using training AICc.", selected_label
    )

    forecast = selected_fit.forecast(len(months))

    return _forecast_series(
        forecast, months, name="ETS"
    )


def _month_dummies(
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Create known calendar-month regressors.

    January is the reference month. All other month indicators
    are generated consistently for training and future dates.
    """
    index = pd.DatetimeIndex(index)

    return pd.DataFrame(
        {
            f"month_{month:02d}": (
                index.month == month
            ).astype(float)
            for month in range(2, 13)
        },
        index=index,
    )


def _sarimax_aicc(
    fitted,
    n_observed: int,
) -> float:
    """Compute a finite-sample AICc for a SARIMAX fit."""
    aic = float(fitted.aic)
    parameter_count = len(fitted.params)

    denominator = n_observed - parameter_count - 1

    if denominator <= 0 or not np.isfinite(aic):
        return float("inf")

    return float(
        aic
        + (
            2 * parameter_count * (parameter_count + 1)
            / denominator
        )
    )


def _fit_best_sarimax(
    train: pd.Series,
    *,
    transform: str,
    use_month_dummies: bool,
):
    """
    Select the order with lowest training AICc.

    Seasonal candidates use (0,1,1)(0,1,1)[12] and
    (0,1,2)(0,1,1)[12].

    Month-dummy candidates use (0,1,1) and (0,1,2) errors
    with D=0. All fitting and order selection use training data.
    """
    if transform not in {"raw", "log"}:
        raise ValueError("transform must be 'raw' or 'log'.")

    endog = _mask_covid(train)

    observed = endog.dropna()

    if len(observed) < 48:
        raise ValueError(
            "Not enough observations for the candidate SARIMA grid."
        )

    if (observed <= 0).any():
        raise ValueError(
            "The sales series must be positive before log fitting."
        )

    if transform == "log":
        endog = np.log(endog)

    exog = (
        _month_dummies(endog.index)
        if use_month_dummies
        else None
    )

    seasonal_order = (
        NO_SEASONAL_DIFFERENCING
        if use_month_dummies
        else SEASONAL_ORDER
    )

    orders = (
        MONTH_DUMMY_ORDERS
        if use_month_dummies
        else SEASONAL_SARIMA_ORDERS
    )

    candidates = []

    for order in orders:
        label = (
            f"SARIMAX order={order}, "
            f"seasonal_order={seasonal_order}, "
            f"scale={transform}"
        )

        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                missing="none",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = model.fit(
                    disp=False,
                    maxiter=300,
                )

            # Do not suppress convergence warnings. Log them explicitly.
            for warning in caught:
                logger.warning(
                    "%s warning (%s): %s",
                    label,
                    warning.category.__name__,
                    warning.message,
                )

            mle_info = getattr(fitted, "mle_retvals", {}) or {}

            if (
                isinstance(mle_info, dict)
                and mle_info.get("converged") is False
            ):
                logger.warning(
                    "%s did not converge.", label
                )

            score = _sarimax_aicc(
                fitted,
                n_observed=len(observed),
            )

            if np.isfinite(score):
                candidates.append(
                    (
                        score,
                        order,
                        seasonal_order,
                        fitted,
                    )
                )
            else:
                logger.warning(
                    "%s returned a non-finite AICc.", label
                )

        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)

    if not candidates:
        raise RuntimeError(
            "All SARIMAX candidates failed. "
            "Review the warnings above."
        )

    score, order, season, fitted = min(
        candidates, key=lambda item: item[0]
    )

    spec = {
        "order": order,
        "seasonal_order": season,
        "scale": transform,
        "month_dummies": use_month_dummies,
        "aicc": float(score),
    }

    logger.info("Selected SARIMAX specification: %s", spec)

    return fitted, spec


def _sarimax_forecast(
    train: pd.Series,
    forecast_months: pd.DatetimeIndex,
    *,
    transform: str,
    use_month_dummies: bool,
    name: str,
) -> pd.Series:
    months = pd.DatetimeIndex(forecast_months)

    fitted, spec = _fit_best_sarimax(
        train,
        transform=transform,
        use_month_dummies=use_month_dummies,
    )

    future_exog = (
        _month_dummies(months)
        if use_month_dummies
        else None
    )

    if future_exog is None:
        predicted = fitted.get_forecast(
            steps=len(months)
        ).predicted_mean
    else:
        predicted = fitted.get_forecast(
            steps=len(months),
            exog=future_exog,
        ).predicted_mean

    predicted = np.asarray(predicted, dtype=float)

    if transform == "log":
        # exp(log forecast) is the conditional median on the
        # dollar scale, not automatically the conditional mean.
        # Bias is assessed later on the original dollar scale.
        predicted = np.exp(predicted)

    return _forecast_series(
        predicted, months, name=name
    )


def sarima_raw_forecast(
    train: pd.Series,
    forecast_months: pd.DatetimeIndex,
) -> pd.Series:
    """Raw-scale seasonal SARIMA, with AICc order selection."""
    return _sarimax_forecast(
        train,
        forecast_months,
        transform="raw",
        use_month_dummies=False,
        name="SARIMA raw",
    )


def sarima_log_forecast(
    train: pd.Series,
    forecast_months: pd.DatetimeIndex,
) -> pd.Series:
    """Log-scale seasonal SARIMA, with AICc order selection."""
    return _sarimax_forecast(
        train,
        forecast_months,
        transform="log",
        use_month_dummies=False,
        name="SARIMA log",
    )


def sarima_month_dummy_log_forecast(
    train: pd.Series,
    forecast_months: pd.DatetimeIndex,
) -> pd.Series:
    """
    Fit SARIMAX on log sales levels with month-dummy regressors.

    This candidate uses ARIMA(0,1,q) errors and no seasonal
    differencing (D=0).

    It is distinct from the Phase 2 OLS regression of log first
    differences on month dummies, which produced R-squared = 0.862.
    That earlier R-squared does not describe this SARIMAX fit.
    """
    return _sarimax_forecast(
        train,
        forecast_months,
        transform="log",
        use_month_dummies=True,
        name="SARIMAX log levels + month dummies (d=1, D=0)",
    )


def fit_sarima_for_diagnostics(
    train: pd.Series,
    *,
    transform: str = "log",
):
    """
    Fit the selected seasonal SARIMA specification for diagnostics.

    Returns (statsmodels_result, selected_specification).
    Use for the training-residual plot and Ljung-Box test.
    """
    return _fit_best_sarimax(
        train,
        transform=transform,
        use_month_dummies=False,
    )


def make_prophet_forecaster(
    seasonality_mode: str,
    changepoint_prior_scale: float,
):
    """
    Return a forecast function for one fixed Prophet configuration.

    Model settings are evaluated using the validation folds.
    No settings are selected using the locked test folds.
    """
    if seasonality_mode not in {"additive", "multiplicative"}:
        raise ValueError(
            "seasonality_mode must be 'additive' or 'multiplicative'."
        )

    if changepoint_prior_scale <= 0:
        raise ValueError(
            "changepoint_prior_scale must be positive."
        )

    # IMPORTANT: forecast_fn must be nested inside this factory.
    def forecast_fn(
        train: pd.Series,
        forecast_months: pd.DatetimeIndex,
    ) -> pd.Series:

        cmdstan_logger = logging.getLogger("cmdstanpy")
        plot_logger = logging.getLogger("prophet.plot")

        previous_cmdstan_disabled = cmdstan_logger.disabled
        previous_plot_disabled = plot_logger.disabled

        # Set these before importing Prophet.
        cmdstan_logger.disabled = True
        plot_logger.disabled = True

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"IProgress not found.*",
                )
                from prophet import Prophet

            # Use the existing Phase 4 COVID training treatment.
            series = _mask_covid(train)

            prophet_train = pd.DataFrame({
                "ds": series.index,
                "y": series.values,
            }).dropna()

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                uncertainty_samples=0,
            )

            model.fit(prophet_train)

            months = pd.DatetimeIndex(forecast_months)
            future = pd.DataFrame({"ds": months})

            result = model.predict(future)

            return _forecast_series(
                result["yhat"].to_numpy(),
                months,
                name=forecast_fn.__name__,
            )

        finally:
            # Restore logger settings even if fitting fails.
            cmdstan_logger.disabled = previous_cmdstan_disabled
            plot_logger.disabled = previous_plot_disabled

    # These lines belong to the factory, outside forecast_fn.
    cps_name = str(changepoint_prior_scale).replace(".", "_")

    forecast_fn.__name__ = (
        f"prophet_{seasonality_mode}_cps_{cps_name}"
    )

    # Return the callable instead of implicitly returning None.
    return forecast_fn





def make_sarima_exog_forecaster(
    macro: pd.DataFrame,
    spec_by_origin: dict,
    feature_set: str | None,
    *,
    name: str | None = None,
    coefficient_records: list | None = None,
    placebo_seed: int | None = None,
):
    """
    Create a forecast_fn compatible with src.evaluation.evaluate().

    The SARIMA order is supplied by spec_by_origin. Within each
    validation fold, every regressor candidate uses the same order
    selected by the no-regressor Phase 4 training fit.

    feature_set:
        None             -> no-regressor reference
        'unemployment'   -> unemployment only
        'cpi'            -> CPI only
        'both'           -> both regressors

    placebo_seed:
        When provided, randomly permute each training regressor
        independently and sample future placebo values from that
        fold's training feature distribution. This never uses
        post-origin macro observations.

    coefficient_records:
        Optional list that receives fold-level coefficient estimates
        and 95% confidence intervals for the genuine regressor models.
    """
    valid_feature_sets = {
        None,
        "unemployment",
        "cpi",
        "both",
    }

    if feature_set not in valid_feature_sets:
        raise ValueError(
            "feature_set must be None, 'unemployment', 'cpi', or 'both'."
        )

    if placebo_seed is not None and feature_set is None:
        raise ValueError(
            "A placebo requires at least one regressor."
        )

    if name is None:
        name = (
            "SARIMA log (no exog)"
            if feature_set is None
            else f"SARIMA log + {feature_set}"
        )

    model_name = name

    def forecast_fn(
        train: pd.Series,
        forecast_months: pd.DatetimeIndex,
    ) -> pd.Series:

        series = _monthly_train(train)
        months = pd.DatetimeIndex(forecast_months)

        origin = series.index.max().to_period("M").to_timestamp()

        if origin not in spec_by_origin:
            raise ValueError(
                f"No training-only SARIMA specification for "
                f"origin {origin:%Y-%m}."
            )

        spec = spec_by_origin[origin]
        order = tuple(spec["order"])
        seasonal_order = tuple(spec["seasonal_order"])

        X_train = None
        X_forecast = None

        if feature_set is not None:
            X_train, X_forecast = build_regressor_matrices(
                macro=macro,
                train_dates=series.index,
                forecast_dates=months,
                origin=origin,
                feature_set=feature_set,
            )

            # The placebo deliberately breaks the historical
            # target-feature alignment. Its values are drawn only
            # from this fold's training regressors.
            if placebo_seed is not None:
                feature_offset = {
                    "unemployment": 101,
                    "cpi": 202,
                    "both": 303,
                }[feature_set]

                rng = np.random.default_rng(
                    placebo_seed
                    + feature_offset
                    + origin.year * 100
                    + origin.month
                )

                X_train = X_train.copy()
                X_forecast = X_forecast.copy()

                for column in X_train.columns:
                    historical_values = (
                        X_train[column].to_numpy(dtype=float).copy()
                    )

                    X_train[column] = rng.permutation(
                        historical_values
                    )

                    X_forecast[column] = rng.choice(
                        historical_values,
                        size=len(X_forecast),
                        replace=True,
                    )

        # Follow Phase 4's training-time COVID treatment.
        # This is not the longer scoring mask.
        endog = series.copy()

        endog.loc[
            COVID_MONTHS.intersection(endog.index)
        ] = np.nan

        observed = endog.dropna()

        if (observed <= 0).any():
            raise ValueError("Sales must be positive before log fitting.")

        log_endog = np.log(endog)

        model = SARIMAX(
            log_endog,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            missing="none",
        )

        # Do not hide convergence or other fitting warnings.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            fitted = model.fit(
                disp=False,
                maxiter=300,
            )

        for item in caught:
            logger.warning(
                "%s, origin %s: warning [%s]: %s",
                model_name,
                origin.strftime("%Y-%m"),
                item.category.__name__,
                item.message,
            )

        mle_info = getattr(fitted, "mle_retvals", {}) or {}

        if (
            isinstance(mle_info, dict)
            and mle_info.get("converged") is False
        ):
            logger.warning(
                "%s did not converge for origin %s.",
                model_name,
                origin.strftime("%Y-%m"),
            )

        # Capture regression coefficients and their fold-level CIs.
        if X_train is not None and coefficient_records is not None:
            parameter_names = list(fitted.model.param_names)
            parameters = np.asarray(fitted.params, dtype=float)
            p_values = np.asarray(fitted.pvalues, dtype=float)
            confidence_intervals = np.asarray(
                fitted.conf_int(alpha=0.05),
                dtype=float,
            )

            for column in X_train.columns:
                if column not in parameter_names:
                    raise KeyError(
                        f"Could not find fitted coefficient {column!r}."
                    )

                position = parameter_names.index(column)
                coefficient = float(parameters[position])
                lower = float(
                    confidence_intervals[position, 0]
                )
                upper = float(
                    confidence_intervals[position, 1]
                )

                coefficient_records.append({
                    "Model": model_name,
                    "Origin": origin.strftime("%Y-%m"),
                    "Regressor": column,
                    "Coefficient": coefficient,
                    "CI_Lower": lower,
                    "CI_Upper": upper,
                    "P_Value": float(p_values[position]),
                    "CI_Includes_Zero": (
                        lower <= 0.0 <= upper
                    ),
                })

        forecast_kwargs = {}

        if X_forecast is not None:
            forecast_kwargs["exog"] = X_forecast

        predicted_log = fitted.get_forecast(
            steps=len(months),
            **forecast_kwargs,
        ).predicted_mean

        # Match Phase 4: return forecasts on the original sales scale.
        # exp(log forecast) is a median forecast, not automatically
        # the conditional mean. Dollar-scale bias remains important.
        predicted = np.exp(
            np.asarray(predicted_log, dtype=float).reshape(-1)
        )

        return _forecast_series(
            predicted,
            months,
            name=model_name,
        )

    forecast_fn.__name__ = model_name

    return forecast_fn
