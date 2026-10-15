import os
from pathlib import Path
import duckdb
import pandas as pd

DB_PATH = Path('data/processed/apparel_forecasting.duckdb')
REPORTS_PATH = Path('reports')
SQL = (Path(__file__).resolve().parents[2] / "sql" / "01-YOY_sales_growth.sql").read_text()

conn = duckdb.connect(DB_PATH.as_posix())

def yoy_sales_growth(file_name):
    REPORT_PATH = REPORTS_PATH / file_name
    
    df = conn.execute(SQL).fetch_df()
    df.to_csv(REPORTS_PATH/file_name, index=False, lineterminator="\n")
    print(f"Exported YOY-Growth-Sales report: {REPORT_PATH.as_posix()} ({len(df)} rows)")
    


if __name__ == "__main__":
    yoy_sales_growth("01-yoy_sales_growth.csv") 