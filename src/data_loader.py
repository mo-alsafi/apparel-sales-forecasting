import os 
import json
from datetime import datetime, timezone
import pandas as pd
import pandas_datareader.data as web


RAW_DIR = "data/raw"
MANIFEST_PATH = os.path.join(RAW_DIR, "retrieval_manifest.json")

SERIES_MAP = {
    "MRTSSM448USN": "apparel_sales_nsa",
    "MRTSSM448USS": "apparel_sales_sa",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate"
}


def ensure_directory():
    os.makedirs(RAW_DIR , exist_ok=True)
    


def execute_data_pull():
    ensure_directory()
        
    manifest = {
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": {} 
    }
    
    for series_id, name in SERIES_MAP.items():  
        file_path = os.path.join(RAW_DIR, f"{series_id}.csv")

        try:
            print(f"Fetching series: {series_id} ({name}) via pandas_datareader...")
            df = web.DataReader(series_id, "fred", start="1990-01-01")
            # Convert Index to explicit 'DATE' column
            df = df.reset_index()
            df.columns = ["DATE", series_id]
            
            # Save clean CSV without dataframe indices
            df.to_csv(file_path, index=False)
            print(f"Successfully saved {series_id}.csv")
        
        except Exception as e:
            if os.path.exists(file_path):
                print(
                    f"Warning: Network download failed for {series_id} ({e})"
                    f"Using existing local file at {file_path}"
                )
                df = pd.DataFrame(file_path)
            else:
                raise RuntimeError(
                    f"Failed to fetch {series_id} and no local cached file exists in {RAW_DIR}."
                ) from e
        
        
        manifest["files"][series_id] = {
            "local_path": file_path,
            "variable_name": name,
            "row_count": len(df),
            "start_date": str(df["DATE"].min()),
            "end_date": str(df["DATE"].max())
        }
        
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=4)
            
        print(f"\nData pull completed successfully. Manifest saved to {MANIFEST_PATH}.")
 


if __name__ == "__main__": 
    execute_data_pull()