import os
from pathlib import Path
import duckdb
import pandas as pd 

DB_PATH = Path("data/processed/apparel_forecasting.duckdb")
REPORTS_PATH = Path("reports")

def best_and_worst_month_per_year(file_name):
    REPORT_PATH = REPORTS_PATH / file_name
    
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("""
        WITH CompleteYears AS (
            SELECT EXTRACT(YEAR FROM observation_date) AS sales_year
            FROM monthly_macro_apparel
            WHERE apparel_sales_nsa IS NOT NULL
            GROUP BY sales_year
            HAVING COUNT(DISTINCT EXTRACT(MONTH FROM observation_date)) = 12
        ),
        RankedMonthlySales AS (
            SELECT 
                EXTRACT(YEAR FROM observation_date) AS sales_year,
                EXTRACT(MONTH FROM observation_date) AS sales_month,
                observation_date,
                apparel_sales_nsa,
                ROW_NUMBER() OVER(
                    PARTITION BY EXTRACT(YEAR FROM observation_date)
                    ORDER BY apparel_sales_nsa DESC
                ) AS rank_highest,
                ROW_NUMBER() OVER (
                    PARTITION BY EXTRACT(YEAR FROM observation_date)
                    ORDER BY apparel_sales_nsa ASC
                ) AS rank_lowest
            FROM monthly_macro_apparel
            WHERE EXTRACT(YEAR FROM observation_date) IN (SELECT sales_year FROM CompleteYears)
        )
        SELECT
            sales_year,
            MAX(CASE WHEN rank_highest = 1 THEN sales_month END) AS peak_month,
            MAX(CASE WHEN rank_highest = 1 THEN apparel_sales_nsa END) AS peak_sales_millions,
            MAX(CASE WHEN rank_lowest = 1 THEN sales_month END) AS trough_month,
            MAX(CASE WHEN rank_lowest = 1 THEN apparel_sales_nsa END) AS trough_sales_millions
        FROM RankedMonthlySales
        GROUP BY sales_year
        ORDER BY sales_year DESC;
    """).fetch_df()
    
    df.to_csv(REPORT_PATH, index=False, lineterminator="\n")
    print(f"Exported Best-and_Worst-Month-per-Year report: {REPORT_PATH.as_posix()} ({len(df)} rows)")


if __name__ == "__main__":
    best_and_worst_month_per_year("03-best_and_worst_month_per_year.csv")