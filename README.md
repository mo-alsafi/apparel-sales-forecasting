# National Apparel Retail Sales Forecasting (12-Month Horizon)

## Business Question
A national apparel retailer requires a 12-month monthly sales forecast with calibrated uncertainty ranges (80% and 95% prediction intervals) to optimize inventory procurement and store staffing schedules.

Which forecasting method should the business trust, under what economic conditions does it hold, and when does it fail?

## Target & Features
- **Target Variable (`y`):** `MRTSSM448USN` — Advance Retail Sales: Clothing and Clothing Accessories Stores (Not Seasonally Adjusted, Millions of USD).
- **Reference Series:** `MRTSSM448USS` — Clothing and Clothing Accessories Stores (Seasonally Adjusted, Millions of USD).
- **Exogenous Regressors:** 
  - `CPIAUCSL` — Consumer Price Index for All Urban Consumers (Inflation proxy).
  - `UNRATE` — Civilian Unemployment Rate (Consumer demand proxy).

## Success Metrics & Baselines
- **Baseline to Beat:** Seasonal Naive Forecast ($y_t = y_{t-12}$).
- **Primary Metric:** MASE (Mean Absolute Scaled Error). Success is defined as $\text{MASE} < 1.0$ on the unseen holdout set.
- **Secondary Metrics:** sMAPE (Symmetric Mean Absolute Percentage Error) and MAE (Mean Absolute Error).

## Validation Strategy
- **Final Test Set:** Last 24 months of historical observations (held out until final model selection).
- **Cross-Validation:** Expanding-window rolling-origin cross-validation with a 12-month forecast horizon across 5 folds.

## Project Structure
- `data/`: Raw API pulls and local DuckDB database.
- `notebooks/`: Exploratory Data Analysis, STL decomposition, and modeling experiments.
- `src/`: Production-ready evaluation routines and data ingestion scripts.
- `reports/`: Model comparison matrices and final business recommendations.