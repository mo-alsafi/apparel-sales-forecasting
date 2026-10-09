import os
from pathlib import Path
import duckdb
import pandas as pd

DB_PATH = Path("data/processed/apparel_forecasting.duckdb")
SQL_DIR = Path("sql")
REPORTS_DIR = Path("reports")

QUERIES = [
    ("01_yoy_growth.sql", "01_yoy_growth.csv"),
    ("02_rolling_12m_avg.sql", "02_rolling_12m_avg.csv"),
    ("03_rank_monthly_sales.sql", "03_rank_monthly_sales.csv"),
]


def execute_sql_runner():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database file not found at {DB_PATH.as_posix()}. "
            "Please run `python src/db_ingestion.py` first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(DB_PATH.as_posix())

    print("Executing SQL screening queries against DuckDB...\n")

    for sql_filename, output_csv_name in QUERIES:
        sql_file_path = SQL_DIR / sql_filename
        output_csv_path = REPORTS_DIR / output_csv_name

        if not sql_file_path.exists():
            print(f"Skipping {sql_filename}: File not found in {SQL_DIR.as_posix()}")
            continue

        with open(sql_file_path, "r", encoding="utf-8") as f:
            query_sql = f.read()

        df_result = conn.execute(query_sql).fetchdf()

        # Save query results with standard Unix line endings
        df_result.to_csv(output_csv_path, index=False, lineterminator="\n")

        print(f"Executed: {sql_filename}")
        print(f"Exported: {output_csv_path.as_posix()} ({len(df_result)} rows)\n")

    conn.close()
    print("All SQL screening reports successfully generated.")


if __name__ == "__main__":
    execute_sql_runner()