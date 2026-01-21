import pandas as pd
from pathlib import Path

def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)