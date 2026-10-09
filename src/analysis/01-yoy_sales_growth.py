import os
from pathlib import Path
import duckdb
import pandas as pd

DB_PATH = Path('data/processed/apparel_forecasting.duckdb')
REPORTS_PATH = Path('reports')

conn = duckdb.connect(DB_PATH.as_posix())

def yoy_sales_growth(file_name):
    REPORT_PATH = REPORTS_PATH / file_name
    
    df = conn.execute("""
        WITH yoy_sales AS (
            SELECT 
                observation_date,
                apparel_sales_nsa,
                LAG(apparel_sales_nsa, 12) OVER (ORDER BY observation_date ASC) AS sales_prior_year
            FROM monthly_macro_apparel
        )
        SELECT 
            observation_date,
            apparel_sales_nsa,
            sales_prior_year,
            ROUND(
                (apparel_sales_nsa - sales_prior_year) / NULLIF(sales_prior_year, 0) * 100, 2
            ) AS yoy_growth_percentage
        FROM yoy_sales
        WHERE observation_date >= '1993-01-01'
            AND apparel_sales_nsa IS NOT NULL
        ORDER BY observation_date ASC;
    """).fetch_df()
    
    df.to_csv(REPORTS_PATH/file_name, index=False, lineterminator="\n")
    print(f"Exported YOY-Growth-Sales report: {REPORT_PATH.as_posix()} ({len(df)} rows)")
    


if __name__ == "__main__":
    yoy_sales_growth("01-yoy_sales_growth.csv") 