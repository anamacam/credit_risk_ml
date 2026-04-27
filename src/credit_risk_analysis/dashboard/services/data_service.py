import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH: Path = Path("logs/predictions.db")


def load_predictions() -> pd.DataFrame:
    """Carga las predicciones desde SQLite y retorna un DataFrame."""

    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        df: pd.DataFrame = pd.read_sql_query(
            "SELECT * FROM model_logs",
            conn
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df
