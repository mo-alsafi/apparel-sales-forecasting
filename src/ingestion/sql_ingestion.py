import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

DATA_MAP = {
    "raw_apparel_sales_nsa": "MRTSSM448USN.csv",
    "raw_apparel_sales_sa": "MRTSSM448USS.csv",
    "raw_cpi": "CPIAUCSL.csv",
    "raw_unemployment_rate": "UNRATE.csv",
}


def raw_data_sql_ingest(data_dir: Path, data_map: dict):
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_pass, db_name]):
        raise ValueError(
            "Missing required database environment variables (DB_USER, DB_PASS, DB_NAME). "
            "Please verify your .env file."
        )

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_pass,
        host=db_host,
        port=int(db_port),
        database=db_name,
    )
    engine = create_engine(connection_url)

    for table_name, file_name in data_map.items():
        file_path = data_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Raw CSV file not found at {file_path.as_posix()}. Run data_loader.py first."
            )

        df = pd.read_csv(file_path)

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
        )
        print(f"Successfully ingested {file_name} into MySQL table '{table_name}' ({len(df)} rows).")


if __name__ == "__main__":
    raw_data_sql_ingest(data_dir=RAW_DATA_DIR, data_map=DATA_MAP)