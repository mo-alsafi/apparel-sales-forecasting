# Apparel Sales Forecasting & Economic Analysis

A data analytics project exploring monthly U.S. apparel retail sales, historical trends, year-over-year growth, seasonality, and the relationship between retail performance and broader economic indicators.

The project combines Python, SQL, and DuckDB to build a reproducible data preparation workflow, conduct exploratory data analysis, and establish a foundation for time-series sales forecasting.

**Project status:** In progress — data ingestion, dataset preparation, and initial SQL analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Business Questions](#business-questions)
- [Data Sources](#data-sources)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Analysis Completed So Far](#analysis-completed-so-far)
- [Data Quality and Validation](#data-quality-and-validation)
- [Getting Started](#getting-started)
- [Current Limitations](#current-limitations)
- [Next Steps](#next-steps)

## Project Overview

Retail sales forecasting helps businesses understand historical demand patterns, anticipate changes in sales, and make better-informed planning decisions.

This project focuses on monthly U.S. apparel retail sales using publicly available economic time-series data from the Federal Reserve Economic Data (FRED) database.

The goal is to develop a structured analytical workflow that transforms raw CSV files into a consolidated dataset suitable for exploratory analysis, statistical investigation, and future forecasting experiments.

Rather than focusing exclusively on building a predictive model, the project emphasizes the data work that precedes forecasting: understanding source data, validating time-series continuity, handling missing observations, exploring trends, and documenting analytical decisions.

## Business Questions

The project is designed to investigate the following questions:

1. How have U.S. apparel retail sales changed over time?
2. What are the year-over-year sales growth rates, and how do they vary across periods?
3. Which months historically record higher or lower apparel sales?
4. How do monthly and longer-term sales trends compare?
5. How do seasonally adjusted sales differ from non-seasonally adjusted sales?
6. What relationships can be observed between apparel sales, consumer prices, and the unemployment rate?
7. How well can historical sales patterns support future monthly sales forecasts?

These are analytical questions, not predetermined conclusions. Findings will be documented after the relevant analysis has been completed.

## Data Sources

The project uses monthly time-series data published by the Federal Reserve Bank of St. Louis through FRED.

| Series ID | Description | Purpose |
|---|---|---|
| `MRTSSM448USN` | Retail sales: Clothing and Clothing Accessories Stores, non-seasonally adjusted | Primary sales series for historical analysis and forecasting |
| `MRTSSM448USS` | Retail sales: Clothing and Clothing Accessories Stores, seasonally adjusted | Comparison and evaluation against the seasonally adjusted series |
| `CPIAUCSL` | Consumer Price Index for All Urban Consumers | Macroeconomic context |
| `UNRATE` | Civilian Unemployment Rate | Labor-market context |

Official data sources:

- [Non-seasonally adjusted apparel retail sales](https://fred.stlouisfed.org/series/MRTSSM448USN)
- [Seasonally adjusted apparel retail sales](https://fred.stlouisfed.org/series/MRTSSM448USS)
- [Consumer Price Index (CPI)](https://fred.stlouisfed.org/series/CPIAUCSL)
- [Unemployment Rate](https://fred.stlouisfed.org/series/UNRATE)

### Target Variable

The primary target is `apparel_sales_nsa`, derived from `MRTSSM448USN`.

The non-seasonally adjusted series preserves the observed monthly seasonal pattern, making it relevant for investigating recurring sales patterns and developing a forecasting baseline.

The seasonally adjusted series is retained as a separate variable for comparison. The two series serve different analytical purposes and should not be treated as interchangeable.

**Data retrieval date:** Record the date on which the source CSV files were downloaded when updating this README.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Pipeline orchestration and data processing |
| Pandas | DataFrames, validation, and CSV export |
| DuckDB | Local analytical database and SQL transformations |
| SQL | Data integration, calendar construction, and exploratory analysis |
| Jupyter Notebook | Interactive exploratory data analysis |
| Git and GitHub | Version control and project documentation |

## Project Structure

The repository is organized to separate source data, ingestion logic, SQL transformations, analysis, and generated outputs.

```text
apparel-sales-forecasting/
│
├── data/
│   ├── raw/                         # Source CSV files
│   └── processed/                   # Local DuckDB database
│
├── reports/
│   └── 00-monthly_macro_apparel.csv  # Consolidated analytical dataset
│
├── sql/
│   └── 00-monthly_macro_apparel.sql  # Dataset construction SQL
│
├── src/
│   ├── ingestion/
│   │   └── db_ingestion.py           # Data ingestion and validation
│   └── eda.py                        # Exploratory analysis scripts
│
├── notebooks/
│   └── 02_eda.ipynb                  # Exploratory data analysis
│
├── requirements.txt
├── .gitignore
└── README.md
```

This structure describes the intended working layout. File names and locations should be updated if the repository changes.

## Data Pipeline

The current ingestion workflow is designed to transform multiple source files into a consolidated monthly dataset.

### 1. Raw Data Ingestion

The pipeline reads the four FRED CSV files from `data/raw/` and creates staging tables in DuckDB:

- `raw_apparel_sales_nsa`
- `raw_apparel_sales_sa`
- `raw_cpi`
- `raw_unemployment_rate`

Dates and numeric observations are cast into appropriate SQL data types. `TRY_CAST` is used for numeric conversion so that unparseable observations can be represented as `NULL`.

### 2. Monthly Calendar Construction

A continuous monthly calendar provides a common date index for joining the four time series.

The calendar is necessary because different indicators can have missing observations or different publication horizons.

### 3. Data Integration and Missing Values

The source series are joined using the monthly observation date to construct the consolidated table:

`monthly_macro_apparel`

The intended output includes:

| Column | Description |
|---|---|
| `observation_date` | Monthly observation date |
| `apparel_sales_nsa` | Non-seasonally adjusted apparel sales |
| `apparel_sales_sa` | Seasonally adjusted apparel sales |
| `cpi` | Consumer Price Index observation |
| `cpi_is_imputed` | Indicator for an interpolated CPI value |
| `unrate` | Unemployment rate |
| `unrate_is_imputed` | Indicator for an interpolated unemployment value |

Missing macroeconomic observations are handled using interpolation where appropriate. Imputation flags are retained to distinguish original observations from estimated values.

Interpolation is not a substitute for investigating missing data. Its suitability depends on the missing observation, the surrounding values, and the intended analytical use.

### 4. Data Validation

The pipeline retrieves the consolidated table into a Pandas DataFrame and runs automated validation assertions.

The checks are intended to verify monthly date continuity, expected imputation counts, sales completeness, and the correctness of the October 2025 CPI interpolation.

The CSV snapshot is exported only after the validation function passes.

### 5. Export

The validated dataset is exported to:

`reports/00-monthly_macro_apparel.csv`

The DuckDB database is stored locally at:

`data/processed/apparel_forecasting.duckdb`

**Implementation note:** The SQL transformation must execute successfully and all validation assertions must pass before the pipeline can be considered operational. These steps are still being verified.

## Analysis Completed So Far

Initial SQL analysis has explored several useful time-series techniques.

### Year-over-Year Sales Growth

Year-over-year growth compares each month's sales with the same month in the previous year.

This comparison helps account for recurring monthly seasonality when evaluating annual changes.

### Moving Average

Moving averages smooth short-term fluctuations to make longer-term changes in sales easier to observe.

The appropriate window depends on the analytical question and should be documented alongside the resulting analysis.

### Monthly Sales Ranking

Window functions, including `ROW_NUMBER()`, have been explored to rank monthly sales observations.

This provides a way to identify high- and low-sales periods and investigate whether the observed rankings reflect recurring seasonal patterns or unusual historical events.

These analyses are exploratory. No quantified business conclusions are reported here until the results have been checked and documented.

## Data Quality and Validation

The project treats data quality as part of the analytical workflow rather than as a separate afterthought.

Current validation objectives include:

- **Temporal continuity:** Confirm that the dataset contains the expected monthly dates.
- **Missing-value handling:** Track the number of imputed observations for each macroeconomic series.
- **Sales completeness:** Verify that the target sales series contains no unexpected missing values within the intended historical analysis period.
- **Interpolation correctness:** Verify that the October 2025 CPI estimate matches the selected interpolation method.
- **Reproducibility:** Generate the consolidated CSV from the source files and SQL transformation.

A validation check passing does not establish that every source value is correct. Source provenance, publication revisions, measurement definitions, and the appropriateness of imputation must also be considered.

## Getting Started

### Prerequisites

- Python installed locally
- Git
- A working virtual environment
- The four source CSV files downloaded from FRED

### 1. Clone the Repository

```bash
git clone https://github.com/mo-alsafi/apparel-sales-forecasting.git
cd apparel-sales-forecasting
```

### 2. Create a Virtual Environment

```bash
python -m venv py_env
```

Activate it on Windows:

```powershell
.\py_env\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate activation command for your shell.

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Source Data

Place the downloaded CSV files in `data/raw/` with these filenames:

```text
data/raw/
├── MRTSSM448USN.csv
├── MRTSSM448USS.csv
├── CPIAUCSL.csv
└── UNRATE.csv
```

The expected column names are `DATE` and the corresponding FRED series ID.

### 5. Run the Ingestion Pipeline

From the repository root:

```bash
python src/ingestion/db_ingestion.py
```

The script reads the SQL transformation, builds the consolidated dataset in DuckDB, runs validation assertions, and exports the CSV snapshot if validation succeeds.

### 6. Explore the Dataset

Once ingestion succeeds, inspect the consolidated CSV or open the exploratory analysis notebook:

```text
notebooks/02_eda.ipynb
```

The analysis should begin with dataset structure, data types, missing values, time-series plots, descriptive statistics, and checks for unusual observations before progressing to deeper analysis.

## Current Limitations

The project is still under development.

- The SQL ingestion workflow and validation checks require successful end-to-end execution.
- Interpolation assumptions need to be documented and tested against the actual source data.
- Exploratory findings have not yet been consolidated into a final analytical report.
- No final forecasting model, benchmark comparison, or out-of-sample accuracy metrics are claimed.
- Relationships between sales and macroeconomic indicators should not be interpreted as causal effects without an appropriate research design.
- Forecast evaluation must respect temporal order and use only information available at each forecast origin.

## Next Steps

The planned development stages are:

1. **Stabilize ingestion:** Resolve SQL execution issues and verify that the complete pipeline runs reproducibly.
2. **Strengthen data validation:** Test duplicate dates, missing periods, unexpected nulls, and source-series coverage.
3. **Complete exploratory data analysis:** Examine distributions, trends, seasonal patterns, year-over-year growth, and unusual observations.
4. **Investigate macroeconomic relationships:** Explore associations between apparel sales, CPI, and unemployment while accounting for trends and seasonality.
5. **Establish forecasting baselines:** Compare simple approaches, such as seasonal naïve forecasting, against suitable time-series models.
6. **Evaluate forecasts:** Use time-based holdouts or rolling-origin validation and report appropriate error metrics.
7. **Document findings:** Translate the analytical results into clear business interpretations and limitations.

## Project Objective

The long-term objective is to build a reproducible, well-documented retail analytics and forecasting project that demonstrates the full analytical process—from source data ingestion and data quality checks to exploratory analysis, forecasting evaluation, and evidence-based interpretation.
