import os
from pathlib import Path
import duckdb
import pandas as pd

DB_PATH = Path("data/processed/apparel_forecasting.duckdb")
REPORTS_DIR = Path("reports")
MASTER_CSV_PATH = REPORTS_DIR / "00-monthly_macro_apparel.csv"


def validate_dataset(df: pd.DataFrame):
    print("Running automated data validation assertions...")

    # 1. Total row count assertion (1990-01-01 to 2026-09-01 = 441 months)
    assert len(df) == 441, f"Validation Failure: Expected 441 rows, found {len(df)}."

    # 2. Date continuity assertion (zero missing months)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    expected_dates = pd.date_range(start="1990-01-01", end="2026-09-01", freq="MS")
    missing_dates = expected_dates.difference(df["observation_date"])
    assert len(missing_dates) == 0, f"Validation Failure: Missing dates detected: {missing_dates}"

    # 3. No NULL sales from 1992-01 to 2026-07
    core_sales = df[
        (df["observation_date"] >= "1992-01-01") & (df["observation_date"] <= "2026-07-01")
    ]["apparel_sales_nsa"]
    null_count = core_sales.isna().sum()
    assert (
        null_count == 0
    ), f"Validation Failure: Found {null_count} NULL sales values in core period (1992-01 to 2026-07)."

    # 4. October 2025 CPI interpolation check
    oct_2025_cpi = df[df["observation_date"] == "2025-10-01"]["cpi"].values[0]
    sep_2025_cpi = df[df["observation_date"] == "2025-09-01"]["cpi"].values[0]
    nov_2025_cpi = df[df["observation_date"] == "2025-11-01"]["cpi"].values[0]

    assert sep_2025_cpi < oct_2025_cpi < nov_2025_cpi, (
        f"Validation Failure: Imputed Oct 2025 CPI ({oct_2025_cpi}) "
        f"does not lie strictly between Sep 2025 ({sep_2025_cpi}) and Nov 2025 ({nov_2025_cpi})."
    )

    print("All 4 data validation checks passed successfully.")


def initialize_database():
    os.makedirs("data/processed", exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DB_PATH.as_posix())

    # 1. Ingest raw CSV files into staging tables
    conn.execute("""
        CREATE OR REPLACE TABLE raw_apparel_sales_nsa AS 
        SELECT 
            CAST(DATE AS DATE) AS "DATE", 
            TRY_CAST(MRTSSM448USN AS DOUBLE) AS apparel_sales_nsa
        FROM read_csv_auto('data/raw/MRTSSM448USN.csv', nullstr=['.', '']);

        CREATE OR REPLACE TABLE raw_apparel_sales_sa AS 
        SELECT 
            CAST(DATE AS DATE) AS "DATE", 
            TRY_CAST(MRTSSM448USS AS DOUBLE) AS apparel_sales_sa
        FROM read_csv_auto('data/raw/MRTSSM448USS.csv', nullstr=['.', '']);

        CREATE OR REPLACE TABLE raw_cpi AS 
        SELECT 
            CAST(DATE AS DATE) AS "DATE", 
            TRY_CAST(CPIAUCSL AS DOUBLE) AS CPIAUCSL
        FROM read_csv_auto('data/raw/CPIAUCSL.csv', nullstr=['.', '']);

        CREATE OR REPLACE TABLE raw_unemployment_rate AS 
        SELECT 
            CAST(DATE AS DATE) AS "DATE", 
            TRY_CAST(UNRATE AS DOUBLE) AS UNRATE
        FROM read_csv_auto('data/raw/UNRATE.csv', nullstr=['.', '']);
    """)

    # 2. Build continuous monthly calendar spine and interpolate October 2025 missing values
    conn.execute("""
        CREATE OR REPLACE TABLE monthly_macro_apparel AS
        WITH date_bounds AS (
            SELECT 
                LEAST(
                    (SELECT MIN("DATE") FROM raw_apparel_sales_nsa),
                    (SELECT MIN("DATE") FROM raw_cpi)
                ) AS min_date,
                GREATEST(
                    (SELECT MAX("DATE") FROM raw_apparel_sales_nsa),
                    (SELECT MAX("DATE") FROM raw_cpi),
                    (SELECT MAX("DATE") FROM raw_unemployment_rate)
                ) AS max_date
        ),
        monthly_spine AS (
            SELECT CAST(generate_series AS DATE) AS observation_date
            FROM date_bounds, generate_series(min_date, max_date, INTERVAL '1 month')
        ),
        raw_joined AS (
            SELECT 
                ms.observation_date,
                nsa.apparel_sales_nsa,
                sa.apparel_sales_sa,
                c.CPIAUCSL AS cpi_raw,
                u.UNRATE AS unrate_raw
            FROM monthly_spine ms
            LEFT JOIN raw_apparel_sales_nsa nsa ON ms.observation_date = nsa."DATE"
            LEFT JOIN raw_apparel_sales_sa sa ON ms.observation_date = sa."DATE"
            LEFT JOIN raw_cpi c ON ms.observation_date = c."DATE"
            LEFT JOIN raw_unemployment_rate u ON ms.observation_date = u."DATE"
        )
        SELECT 
            observation_date,
            apparel_sales_nsa,
            apparel_sales_sa,
            
            -- CPI Interpolation
            CASE 
                WHEN cpi_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_cpi) THEN
                    (LAG(cpi_raw, 1) OVER (ORDER BY observation_date) + LEAD(cpi_raw, 1) OVER (ORDER BY observation_date)) / 2.0
                ELSE cpi_raw
            END AS cpi,
            
            CASE 
                WHEN cpi_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_cpi) THEN 1 
                ELSE 0 
            END AS cpi_is_imputed,

            -- Unemployment Interpolation
            CASE 
                WHEN unrate_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_unemployment_rate) THEN
                    (LAG(unrate_raw, 1) OVER (ORDER BY observation_date) + LEAD(unrate_raw, 1) OVER (ORDER BY observation_date)) / 2.0
                ELSE unrate_raw
            END AS unrate,
            
            CASE 
                WHEN unrate_raw IS NULL AND observation_date < (SELECT MAX("DATE") FROM raw_unemployment_rate) THEN 1 
                ELSE 0 
            END AS unrate_is_computed

        FROM raw_joined
        ORDER BY observation_date ASC;
    """)

    # 3. Export master table to reports/00-monthly_macro_apparel.csv for database-free modeling
    df_master = conn.execute("SELECT * FROM monthly_macro_apparel").fetchdf()
    df_master.to_csv(MASTER_CSV_PATH, index=False, lineterminator="\n")
    print(f"Exported master dataset: {MASTER_CSV_PATH.as_posix()} ({len(df_master)} rows)")

    # 4. Run automated validation assertions
    validate_dataset(df_master)

    conn.close()


if __name__ == "__main__":
    initialize_database()