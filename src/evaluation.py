import pandas as pd
import numpy as np

def get_forecast_folds(origins, horizon=12):
    """
    Generates training cutoff dates and forecast month ranges for a list of origin dates.
    Origins are typically the end of July (e.g., 2016-07-31).
    """
    folds = []
    for origin in origins:
        origin_ts = pd.to_datetime(origin)
        # Forecast months span the next 12 months starting August
        forecast_months = pd.date_range(
            start=origin_ts + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS"
        )
        folds.append({
            "origin": origin_ts,
            "train_end": origin_ts,
            "forecast_months": forecast_months
        })
    return folds

def apply_scoring_mask(y_true, y_pred, mask_start="2020-03-01", mask_end="2021-06-30"):
    """
    Applies the COVID scoring mask to exclude pandemic shock and rebound windows 
    from quantitative error metrics.
    """
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas Series with a DateTime index.")
        
    mask = ~((y_true.index >= mask_start) & (y_true.index <= mask_end))
    
    if isinstance(y_pred, pd.Series):
        return y_true[mask], y_pred[mask]
    else:
        # If y_pred is a numpy array matching y_true length
        return y_true[mask], y_pred[mask[mask.index == y_true.index]] # handled via series alignment

def calculate_mae(y_true, y_pred):
    """Calculates Mean Absolute Error ($ Millions)."""
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error ($ Millions)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_bias(y_true, y_pred):
    """Calculates Mean Error / Bias. Positive indicates over-forecasting."""
    return float(np.mean(y_pred - y_true))

def calculate_rel_mae(y_true, y_pred, y_pred_baseline):
    """
    Calculates Relative Mean Absolute Error (RelMAE) relative to a benchmark model 
    (e.g., Seasonal Naive). Target is <= 0.90.
    """
    mae_model = calculate_mae(y_true, y_pred)
    mae_baseline = calculate_mae(y_true, y_pred_baseline)
    
    if mae_baseline == 0:
        return np.nan
        
    return float(mae_model / mae_baseline)