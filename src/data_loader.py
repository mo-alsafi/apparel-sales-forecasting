import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pandas_datareader.data as web

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "retrieval_manifest.json"

SERIES_MAP = {
    "MRTSSM448USN": "apparel_sales_raw",
    "MRTSSM448USS": "apparel_sales_sa",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate",
}


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def execute_data_pull():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    for series_id, name in SERIES_MAP.items():
        file_path = RAW_DIR / f"{series_id}.csv"
        source = "fred_live"

        try:
            print(f"Fetching series: {series_id} ({name}) via pandas_datareader...")
            df = web.DataReader(series_id, "fred", start="1990-01-01")
            df = df.reset_index()
            df.columns = ["DATE", series_id]
            
            # Write with explicit LF line endings for cross-platform SHA-256 consistency
            df.to_csv(file_path, index=False, lineterminator="\n")
            print(f"Successfully downloaded and saved {series_id}.csv")

        except Exception as e:
            if file_path.exists():
                print(
                    f"Warning: Network request failed for {series_id} ({e}). "
                    f"Falling back to local cached snapshot at {file_path.as_posix()}."
                )
                source = "local_cache"
                df = pd.read_csv(file_path, parse_dates=["DATE"])
            else:
                raise RuntimeError(
                    f"Failed to fetch {series_id} and no local snapshot exists in {RAW_DIR.as_posix()}."
                ) from e

        date_series = pd.to_datetime(df["DATE"])

        manifest["files"][series_id] = {
            "local_path": file_path.as_posix(),
            "variable_name": name,
            "source": source,
            "sha256": compute_sha256(file_path),
            "row_count": len(df),
            "start_date": date_series.min().strftime("%Y-%m-%d"),
            "end_date": date_series.max().strftime("%Y-%m-%d"),
        }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nData pull completed successfully. Manifest saved to {MANIFEST_PATH.as_posix()}")


if __name__ == "__main__":
    execute_data_pull()