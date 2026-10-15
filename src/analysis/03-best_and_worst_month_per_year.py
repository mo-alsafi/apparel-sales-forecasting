import os
from pathlib import Path
import duckdb
import pandas as pd 

DB_PATH = Path("data/processed/apparel_forecasting.duckdb")
REPORTS_PATH = Path("reports")
SQL = (Path(__file__).resolve().parents[2] / "sql" / "03-best_and_worst_month_per_year.sql").read_text()

def best_and_worst_month_per_year(file_name):
    REPORT_PATH = REPORTS_PATH / file_name
    
    conn = duckdb.connect(DB_PATH)
    df = conn.execute(SQL).fetch_df()
    
    df.to_csv(REPORT_PATH, index=False, lineterminator="\n")
    print(f"Exported Best-and_Worst-Month-per-Year report: {REPORT_PATH.as_posix()} ({len(df)} rows)")


if __name__ == "__main__":
    best_and_worst_month_per_year("03-best_and_worst_month_per_year.csv")