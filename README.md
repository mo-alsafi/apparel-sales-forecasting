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
- **Pipeline Processing:** Raw CSV files in `data/raw/` are preserved untouched. The processing layer in `src/ingestion/db_ingestion.py` performs linear interpolation for this single-month gap:
  - **CPI Interpolation:** $\frac{\text{CPI}_{\text{Sep25}} + \text{CPI}_{\text{Nov25}}}{2} = \frac{324.245 + 325.063}{2} = 324.654$
  - **UNRATE Interpolation:** $\frac{\text{UNRATE}_{\text{Sep25}} + \text{UNRATE}_{\text{Nov25}}}{2} = \frac{4.4 + 4.5}{2} = 4.45$
- **Imputation Audit:** Explicit boolean indicators (`cpi_is_imputed` and `unrate_is_computed`) are stored in `monthly_macro_apparel` to track transformed values.

### 3. Exogenous Alignment & Date Spine Architecture
- **Calendar Spine:** The target table `monthly_macro_apparel` is constructed using an explicit monthly date spine (`generate_series`). This prevents `LEFT JOIN` operations on sales from truncating available macro regressors when CPI or UNRATE data extend past the latest published sales month.

### 4. Future Exogenous Availability
- **The Information Leakage Catch:** At forecast origin $t$, future exogenous regressors ($CPI_{t+h}$ and $UNRATE_{t+h}$) for horizon $h \in [1, 12]$ are unobserved.
- **Handling Strategies:**
  1. **Lagged Feature Architecture:** Use $h$-step lagged values ($X_{t}$) so predictions depend only on known historical macro data.
  2. **Univariate Regressor Forecasting:** Fit independent ARIMA models to project CPI and Unemployment 12 months ahead before feeding into SARIMAX/Prophet.
  3. **Macroeconomic Scenario Analysis:** Evaluate Base, Optimistic, and Stress scenarios for exogenous inputs.

---

## EDA Findings (Phase 2)

Full analysis: [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb) (helpers in `src/eda.py`). Sample: 1992-01 to 2026-07, 415 monthly observations of NSA sales. Each figure is listed with what it means for the model.

| Figure | What it shows | So what |
| :--- | :--- | :--- |
| `fig1_nsa_vs_sa_overview.png` | NSA vs SA sales, 1992–2026 | Seasonality dominates (every December is about twice a January), so models are fitted on the NSA series with explicit seasonality. After the 2020 dip, sales settled at a **higher level** instead of returning to the old path. |
| `fig2_seasonality_and_dec_jan_ratio.png` | Seasonal index by month; December/January ratio by year | The seasonal pattern is **flattening**, not fixed: the Dec/Jan ratio averaged 2.51 (1992–99), 2.34 (2000–06) and about 2.1 from 2007 on (trend ≈ −0.016 per year). Fixed seasonal factors are not appropriate; use seasonal differencing or flexible seasonality. |
| `fig3_growth_regimes_yoy.png` | YoY growth (axis clipped to −20…+30%) and recovery vs 2019 | 2008–09: YoY was negative in 16 of 24 months (mean −3.5%, low −12.5% in Dec 2008). COVID: −87% YoY in April 2020. The 2021 YoY spike (+751% in April) is a **base effect**; against 2019, 2021 was +8.2% nominal, about +2.1% real. In 2022 nominal sales grew +3.7% while CPI rose +8.0%, so **real sales fell about 3.9%**. |
| `fig4_log_transformation_variance.png` | Level, log level, and seasonal swing vs sales level | The seasonal swing is only **partly** proportional to the level (slope b = 0.57, Box-Cox λ ≈ 0.43). Log is the working default, not a settled fact: the transform is compared by CV in Phase 5. |
| `fig5_stl_decomposition.png` | Robust STL on log sales | Trend, seasonal and residual separate cleanly. The four largest residuals are the COVID months March–June 2020; nothing else in 34 years is comparable, so COVID needs an explicit treatment. |
| `fig6_acf_pacf_diagnostics.png` | ACF/PACF of the (d=1, D=1) log series, full vs pre-2020 | COVID distorts the autocorrelations (full-sample ACF lag 2 = −0.41 comes from April 2020). The pre-2020 sample shows the airline-model pattern (ACF lag 1 = −0.54, lag 12 = −0.27), so orders are identified on **pre-2020 data only**. |
| `fig7_sales_vs_macro.png` | Sales growth vs CPI inflation and unemployment | Monthly changes carry no usable signal. Year-over-year changes co-move moderately and contemporaneously (unemployment r ≈ −0.44, CPI r ≈ +0.24 with COVID excluded), and the relationship fades with lag. |

Additional statistics:
- **Seasonal amplitude:** from 1992–99 to 2016–19 the average sales level rose 94% (11,053 → 21,447 $M) but the within-year standard deviation rose only 42% (3,104 → 4,401 $M); the standard deviation of log sales *fell* from 0.237 to 0.185.
- **COVID shock size:** April 2020 was −85.8% vs February (naive), −87.8% vs the usual seasonal ratio, −87.4% year-over-year.
- **Macro caveat:** YoY series overlap by 11 months, so correlations are indicative only and the usual ±1.96/√n band does not apply. CPI is the all-items index, not an apparel price index.
- **Not tested:** *why* things happened (stimulus, online shopping, inflation) is hypothesis, not a result of this analysis.

---

## COVID-19 Treatment Decision

COVID-19 is a once-in-a-century shock that would otherwise corrupt identification, fitting and scoring. The decision, applied to **every** model and baseline:

1. **Identify** SARIMA orders and differencing on the pre-2020 sample (1992-01 to 2019-12, 336 months).
2. **Fit** on the full history, but **mask March–June 2020 as missing** (NaN; neither zero nor interpolated). State-space SARIMAX and Prophet accept missing values.
3. **Score** without the affected targets: forecast targets in 2020-03..06 *and* 2021-03..06 are excluded from all metrics, including the MASE scale. The 2021 months are excluded because their seasonal-naive reference (t−12) is a COVID month.
4. **Sensitivity checks (Phases 5–6):** intervention dummies for Mar–Jun 2020, pre-2020-only training, post-2021-only training. Conclusions are reported with and without COVID.
5. **Level shift:** masking does not remove the post-2021 level shift (SA sales averaged $21,747M in 2019 and $24,417M in 2022, +12%). Differencing absorbs a one-off shift; Prophet gets a changepoint check.

---

## Modeling Implications (carried into Phases 3–7)

| Decision | Choice | Evidence |
| :--- | :--- | :--- |
| Target | NSA sales, $M | Seasonality is modeled explicitly (fig1) |
| Transform | Log as working default; treated as a hyperparameter (none vs log vs Box-Cox λ ≈ 0.4) | Partly multiplicative, b = 0.57 (fig4) |
| Differencing | d = 1, D = 1 | Level and seasonal non-stationarity (stationarity table, STL seasonal strength) |
| SARIMA orders | Primary: (0,1,1)(0,1,1)₁₂; compared by AICc and CV: (1,1,1)(0,1,1)₁₂ and (0,1,2)(0,1,1)₁₂ | ACF/PACF on pre-2020 data (fig6) |
| Seasonality | Allow it to drift; no fixed seasonal factors | Dec/Jan ratio falls from ~2.5 to ~2.1 (fig2) |
| COVID | Mask on fit, exclude from scoring, run sensitivity checks | See the decision above |
| Exogenous regressors | CPI / UNRATE as lagged YoY terms, tested out of sample in Phase 7 | Moderate contemporaneous YoY co-movement that fades with lag (fig7) |

---

## Validation Strategy

- **Final Test Set:** 24-month untouched holdout window.
- **Cross-Validation Scheme:** Expanding window rolling-origin CV (12-month horizon, 5 folds).
- **COVID-aware scoring:** forecast targets in 2020-03..06 and 2021-03..06 are excluded from every metric for every model and for the seasonal-naive baseline (see the COVID decision above).