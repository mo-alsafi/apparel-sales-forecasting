from pathlib import Path
import pandas as pd

# Anchor to the repository root so this works from any working directory
# (notebooks/, the repo root, pytest, scripts).
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_apparel_data(filename="00-monthly_macro_apparel.csv"):
    DATA_PATH = REPO_ROOT / "reports" / filename

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    df = df.loc["1992-01-01": "2026-07-01"].copy()
    df = df.asfreq("MS")

    nsa = "apparel_sales_nsa"
    if nsa in df.columns:
        assert df[nsa].isna().sum() == 0, f"Found NaN values in target column {nsa}"

    return df
