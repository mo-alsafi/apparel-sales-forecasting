from pathlib import Path
import pandas as pd 


def load_apparel_data(filename="00-monthly_macro_apparel.csv"):
    DATA_PATH = Path.cwd().parent / "reports" /  filename
    
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
        assert df[nsa].isna().sum() == 0, f"Found NaN values in in target columns {nsa}"
        
    return df