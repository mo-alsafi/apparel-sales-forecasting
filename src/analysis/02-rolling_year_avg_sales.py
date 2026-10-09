import os
from pathlib import Path
import duckdb
import pandas as pd 

DB_PATH = Path("data/processed/apparel_forecasting.duckdb")
REPORTS_PATH = Path("reports")

def rolling_year_avg_sales(file_name):
    REPORT_PATH = REPORTS_PATH / file_name
    
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("""
        WITH rolling_year_avg_sales AS(
            SELECT
                observation_date,
                apparel_sales_nsa,
                ROUND(
                    AVG(apparel_sales_nsa) OVER (
                        ORDER BY observation_date ASC
                        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                    ), 2
                ) AS rolling_yearly_avg_sales
            FROM monthly_macro_apparel
        )
        SELECT
            observation_date,
            apparel_sales_nsa,
            rolling_yearly_avg_sales
        FROM rolling_year_avg_sales
        WHERE observation_date >= '1993-01-01'
            AND apparel_sales_nsa IS NOT NULL
        ORDER BY observation_date ASC;
    """).fetch_df()
    
    df.to_csv(REPORT_PATH, index=False, lineterminator="\n")
    print(f"Exported Rolling-Year_Avg_Sales report: {REPORT_PATH.as_posix()} ({len(df)} rows)")


if __name__ == "__main__":
    rolling_year_avg_sales("02-rolling_year_avg_sales.csv")