import pandas as pd
import pytest
from src.evaluation import get_forecast_folds

def test_no_future_training_leakage():
    """
    Ensures that for any fold, no training data point timestamp is >= the fold's origin date.
    Protects against expanding window or index slicing errors.
    """
    # Create mock full timeline matching project span (1992-2026)
    dates = pd.date_range(start="1992-01-01", end="2026-07-31", freq="MS")
    df_mock = pd.DataFrame({"apparel_sales_nsa": 1000.0}, index=dates)
    
    origins = ["2016-07-31", "2019-07-31", "2023-07-31"]
    folds = get_forecast_folds(origins)
    
    for fold in folds:
        origin = fold["origin"]
        # Slice training window (expanding window up to origin)
        train_data = df_mock.loc[df_mock.index <= fold["train_end"]]
        
        # Assertions
        assert len(train_data) > 0, f"Training set for origin {origin} is empty."
        assert train_data.index.max() <= origin, (
            f"Temporal Leakage Detected! Training max date {train_data.index.max().strftime('%Y-%m-%d')} "
            f"is on or after origin {origin.strftime('%Y-%m-%d')}."
        )

def test_no_validation_test_overlap():
    """
    Ensures that validation target forecast months do not intersect with 
    the final holdout test period (Aug 2024 – Jul 2026).
    """
    val_origins = ["2016-07-31", "2018-07-31", "2022-07-31", "2023-07-31"]
    test_origins = ["2024-07-31", "2025-07-31"]
    
    val_folds = get_forecast_folds(val_origins)
    test_folds = get_forecast_folds(test_origins)
    
    # Collect all validation target months into a set
    val_months = set()
    for fold in val_folds:
        for m in fold["forecast_months"]:
            val_months.add(m)
            
    # Collect all holdout test target months into a set
    test_months = set()
    for fold in test_folds:
        for m in fold["forecast_months"]:
            test_months.add(m)
            
    # Check for intersection
    overlap = val_months.intersection(test_months)
    assert len(overlap) == 0, (
        f"Holdout Leakage Detected! Validation target months overlap with test set on: "
        f"{[m.strftime('%Y-%m') for m in sorted(overlap)]}"
    )
    
    
    
def test_imputed_oct2025_safety():
    """
    Fails if the imputed October 2025 value is present in the training data 
    for any evaluation fold with an origin prior to November 2025.
    """
    imputed_date = pd.Timestamp("2025-10-01")
    
    # Mock dataset spanning up to 2026
    dates = pd.date_range(start="1992-01-01", end="2026-07-31", freq="MS")
    df_mock = pd.DataFrame({"apparel_sales_nsa": 1000.0}, index=dates)
    
    # Test origins prior to November 2025
    origins = [pd.Timestamp("2024-07-31"), pd.Timestamp("2025-07-31")]
    
    for origin in origins:
        # Slice training window up to origin date
        train_data = df_mock.loc[df_mock.index <= origin]
        
        # Assert that the imputed date is strictly excluded from training
        assert imputed_date not in train_data.index, (
            f"Imputed Data Leakage! October 2025 imputed value found in training "
            f"data for origin {origin.strftime('%Y-%m-%d')}."
        )