# National Apparel Retail Sales Forecasting (12-Month Horizon)

## Business Question
A national apparel retailer requires a 12-month monthly sales forecast with calibrated uncertainty ranges (80% and 95% prediction intervals) to optimize inventory procurement and store staffing schedules.

Which forecasting method should the business trust, under what economic conditions does it hold, and when does it fail?

---

## Success Metrics & Baselines

- **Primary Benchmark (Relative MAE):** 
  $$\text{RelMAE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{Seasonal Naive}}}$$
  Success is defined as $\text{RelMAE} < 1.0$ evaluated on identical forecast horizons across all rolling cross-validation folds and the holdout test set.
- **Secondary Metrics:**
  - **MASE (Mean Absolute Scaled Error):** Evaluates error scaled against in-sample seasonal naive baseline error.
  - **sMAPE (Symmetric Mean Absolute Percentage Error):** Measures percentage error bounded between 0% and 200%.
  - **MAE (Mean Absolute Error):** Expressed in millions of USD for direct operational interpretation.

---

## Data Dictionary

| Variable Name | Series ID | Official Source | Units | Frequency | Seasonality | Revision Policy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `apparel_sales_raw` | `MRTSSM448USN` | U.S. Census Bureau (MRTS) | Millions of USD | Monthly | Not Seasonally Adjusted | Revised monthly; full annual benchmark update in April |
| `apparel_sales_sa` | `MRTSSM448USS` | U.S. Census Bureau (MRTS) | Millions of USD | Monthly | Seasonally Adjusted | Revised backward 3–5 years during seasonal adjustment cycles |
| `cpi` | `CPIAUCSL` | U.S. Bureau of Labor Statistics | Index 1982–1984=100 | Monthly | Seasonally Adjusted | Revised annually for seasonal factors (last 5 years) |
| `unemployment_rate` | `UNRATE` | U.S. Bureau of Labor Statistics | Percent (%) | Monthly | Seasonally Adjusted | Subject to annual historical benchmark revisions |

*Note: `MRTSSM448USN` represents the full Monthly Retail Trade Survey (MRTS) for Clothing and Clothing Accessories Stores, whereas `RSCCASN` represents Advance Retail Sales.*

---

## Data Notes & Caveats

### 1. Retrieval Snapshot & Revisions
- **Data Pull Date:** October 2026
- **Revision Risk:** Economic time series from FRED are subject to historical revisions. The standard deviation of revisions is highest in the most recent 3–6 months. Models trained on unrevised historical data will evaluate against revised series in real-world deployment.

### 2. Missing Data & October 2025 CPI Gap
- **October 2025 CPI Anomaly:** The October 2025 observation for `CPIAUCSL` exhibits a missing/delayed release value in public FRED tables.
- **Remediation Strategy:** Imputed via linear interpolation using adjacent month observations ($\frac{\text{CPI}_{\text{Sep}} + \text{CPI}_{\text{Nov}}}{2}$) prior to model ingestion. `UNRATE` verified complete across all periods.

### 3. Exogenous Variable Forecast Alignment
- **The Information Leakage Catch:** At forecast origin $t$, future exogenous regressors ($CPI_{t+h}$ and $UNRATE_{t+h}$) for horizon $h \in [1, 12]$ are unobserved.
- **Handling Strategies:**
  1. **Lagged Feature Architecture:** Use $h$-step lagged values ($X_{t}$) so predictions depend only on known historical macro data.
  2. **Univariate Regressor Forecasting:** Fit independent ARIMA models to project CPI and Unemployment 12 months ahead before feeding into SARIMAX/Prophet.
  3. **Macroeconomic Scenario Analysis:** Evaluate Base, Optimistic, and Stress scenarios for exogenous inputs.

### 4. Cross-Validation Structure & COVID-19 Treatment
- **Holdout Test Set:** Last 24 months of observations held out until final model evaluation.
- **Rolling-Origin CV:** 5 expanding window folds with a 12-month horizon.
- **COVID-19 Structural Shock (2020):**
  - March–May 2020 contains extreme negative outliers due to retail store closures.
  - **Mitigation:** Evaluated using (a) Intervention Dummy Variables (March–June 2020 = 1), (b) Pre-2020 vs. Post-2020 training windows, and (c) Outlier adjustment via STL decomposition residuals.

---

## Validation Strategy

- **Final Test Set:** 24-month untouched holdout window.
- **Cross-Validation Scheme:** Expanding window rolling-origin CV (12-month horizon, 5 folds).