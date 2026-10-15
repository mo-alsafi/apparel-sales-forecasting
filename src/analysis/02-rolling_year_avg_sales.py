import os
from pathlib import Path
import duckdb
import pandas as pd 

DB_PATH = Path("data/processed/apparel_forecasting.duckdb")
REPORTS_PATH = Path("reports")
SQL = Path(__file__).resolve().parents[2] / "sql" / "02-rolling_year_avg_sales.csv".read_text()

def rolling_year_avg_sales(file_name):
    REPORT_PATH = REPORTS_PATH / file_name
    
    conn = duckdb.connect(DB_PATH)
    df = conn.execute(SQL).fetch_df()
    
    df.to_csv(REPORT_PATH, index=False, lineterminator="\n")
    print(f"Exported Rolling-Year_Avg_Sales report: {REPORT_PATH.as_posix()} ({len(df)} rows)")


if __name__ == "__main__":
    rolling_year_avg_sales("02-rolling_year_avg_sales.csv")