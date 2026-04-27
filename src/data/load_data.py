from pathlib import Path
from typing import Set
import pandas as pd

# ==============================
# CONFIG
# ==============================

RAW_DATA_PATH: Path = Path("data/raw/german_credit_data.csv")
PROCESSED_DATA_PATH: Path = Path("data/processed/loaded_data.csv")

EXPECTED_COLUMNS: Set[str] = {
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
    "Risk",
}


# ==============================
# FUNCTIONS
# ==============================


def load_data(path: Path) -> pd.DataFrame:
    """Carga el dataset desde CSV con validación de ruta."""
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    df: pd.DataFrame = pd.read_csv(path)
    print(f"\n✅ Datos cargados. Shape: {df.shape}")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Verifica que existan las columnas esperadas.
    Resuelve no-untyped-def [cite: 2026-03-04].
    """
    cols: Set[str] = set(df.columns)

    missing: Set[str] = EXPECTED_COLUMNS - cols
    if missing:
        raise ValueError(f"Faltan columnas críticas: {missing}")

    print("✅ Validación de columnas exitosa")


def basic_report(df: pd.DataFrame) -> None:
    """Genera un reporte rápido de calidad de datos."""
    print("\n===== INFORMACIÓN BÁSICA =====")
    df.info()

    print("\n===== VALORES NULOS =====")
    print(df.isnull().sum())

    print("\n===== DUPLICADOS =====")
    print(df.duplicated().sum())


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Guarda una copia procesada en la ruta especificada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n✅ Guardado en: {path}")


# ==============================
# MAIN
# ==============================


def main() -> None:
    """Punto de entrada con tipado estricto."""
    df: pd.DataFrame = load_data(RAW_DATA_PATH)

    validate_columns(df)
    basic_report(df)
    save_processed(df, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
