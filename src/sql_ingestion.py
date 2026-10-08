import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

DATA_MAP = {
    "raw_apparel_sales_nsa": "MRTSSM448USN.csv",
    "raw_apparel_sales_sa": "MRTSSM448USS.csv",
    "raw_cpi": "CPIAUCSL.csv",
    "raw_unemployment_rate": "UNRATE.csv"
}

def raw_data_sql_ingest(DATA_DIR, DATA_MAP):
    DB_USER, DB_PASS = os.getenv("DB_USER"), os.getenv("DB_PASS")
    DB_HOST, DB_PORT, DB_NAME = os.getenv("DB_HOST"), os.getenv("DB_PORT"), os.getenv("DB_NAME")
    
    conn_string = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_string)
    
    for dname, fname in DATA_MAP.items():
        data_path = DATA_DIR / fname
        df = pd.read_csv(data_path)
        
        try:
            df.to_sql(
                name=dname,
                if_exists="replace",
                con=engine,
                index=False
            )
            print(f"##### Done ingesting file {fname} ({dname}) to Mysql.")
        except Exception as e:
            print(f"XXXXX Couldn't ingest file {fname} ({dname}) to Mysql here is why: \n {e}")
            

if __name__ == "__main__":
    raw_data_sql_ingest(DATA_DIR=RAW_DATA_DIR, DATA_MAP=DATA_MAP)