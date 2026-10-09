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
| `apparel_sales_nsa` | `MRTSSM448USN` | U.S. Census Bureau (MRTS) | Millions of USD | Monthly | Not Seasonally Adjusted | Revised monthly; full annual benchmark update in April |
| `apparel_sales_sa` | `MRTSSM448USS` | U.S. Census Bureau (MRTS) | Millions of USD | Monthly | Seasonally Adjusted | Revised backward 3–5 years during seasonal adjustment cycles |
| `cpi` | `CPIAUCSL` | U.S. Bureau of Labor Statistics | Index 1982–1984=100 | Monthly | Seasonally Adjusted | Revised annually for seasonal factors (last 5 years) |
| `unrate` | `UNRATE` | U.S. Bureau of Labor Statistics | Percent (%) | Monthly | Seasonally Adjusted | Subject to annual historical benchmark revisions |

*Note: `MRTSSM448USN` represents the full Monthly Retail Trade Survey (MRTS) for Clothing and Clothing Accessories Stores, whereas `RSCCASN` represents Advance Retail Sales.*

---

## Data Notes & Caveats

### 1. Retrieval Snapshot & Revisions
- **Data Pull Date:** October 2026
- **Revision Risk:** Economic time series from FRED are subject to historical revisions. The standard deviation of revisions is highest in the most recent 3–6 months. Models trained on unrevised historical data will evaluate against revised series in real-world deployment.

### 2. Missing Data & October 2025 Gap Imputation
- **October 2025 Anomaly:** Both `CPIAUCSL` and `UNRATE` contain missing values for the `2025-10-01` release in FRED snapshots.
- **Pipeline Processing:** Raw CSV files in `data/raw/` are preserved untouched. The processing layer in `src/db_ingestion.py` performs linear interpolation for this single-month gap:
  - **CPI Interpolation:** $\frac{\text{CPI}_{\text{Sep25}} + \text{CPI}_{\text{Nov25}}}{2} = \frac{324.245 + 325.063}{2} = 324.654$
  - **UNRATE Interpolation:** $\frac{\text{UNRATE}_{\text{Sep25}} + \text{UNRATE}_{\text{Nov25}}}{2} = \frac{4.4 + 4.5}{2} = 4.45$
- **Imputation Audit:** Explicit boolean indicators (`cpi_is_imputed` and `unrate_is_imputed`) are stored in `monthly_macro_apparel` to track transformed values.

### 3. Exogenous Alignment & Date Spine Architecture
- **Calendar Spine:** The target table `monthly_macro_apparel` is constructed using an explicit monthly date spine (`generate_series`). This prevents `LEFT JOIN` operations on sales from truncating available macro regressors when CPI or UNRATE data extend past the latest published sales month.

### 4. Future Exogenous Availability
- **The Information Leakage Catch:** At forecast origin $t$, future exogenous regressors ($CPI_{t+h}$ and $UNRATE_{t+h}$) for horizon $h \in [1, 12]$ are unobserved.
- **Handling Strategies:**
  1. **Lagged Feature Architecture:** Use $h$-step lagged values ($X_{t}$) so predictions depend only on known historical macro data.
  2. **Univariate Regressor Forecasting:** Fit independent ARIMA models to project CPI and Unemployment 12 months ahead before feeding into SARIMAX/Prophet.
  3. **Macroeconomic Scenario Analysis:** Evaluate Base, Optimistic, and Stress scenarios for exogenous inputs.

---

## Validation Strategy

- **Final Test Set:** 24-month untouched holdout window.
- **Cross-Validation Scheme:** Expanding window rolling-origin CV (12-month horizon, 5 folds).