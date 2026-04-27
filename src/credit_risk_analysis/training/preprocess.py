from __future__ import annotations

import os
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("preprocess_service")

# --- Lógica de Rutas Agnóstica (Docker-First) ---
if os.name != 'nt':
    # Entorno Linux / Docker
    PROJECT_ROOT = Path("/app")
else:
    # Entorno Local Windows (Fallback)
    DEFAULT_WIN = "C:/projects/credit_risk_ml"
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", DEFAULT_WIN))

# Paths normalizados
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "german_credit_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache"

# Asegurar directorios
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.warning(f"No se pudo crear infraestructura de carpetas: {e}")


def load_data(path: Union[Path, str]) -> pd.DataFrame:
    """Carga defensiva con resolución de paths para Docker."""
    target_path = Path(path)

    if not target_path.exists():
        docker_path = Path("/app/data/raw/german_credit_data.csv")
        if docker_path.exists():
            target_path = docker_path
            logger.info(f"Usando fallback de Docker: {target_path}")
        else:
            logger.critical(f"Dataset no encontrado en {target_path}")
            raise FileNotFoundError(f"Falta dataset: {target_path}")

    try:
        df: pd.DataFrame = pd.read_csv(target_path)
        if df.empty:
            raise ValueError(f"El dataset en {target_path} está vacío.")

        df = df.drop_duplicates()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        return df
    except Exception as e:
        logger.error(f"Error fatal en lectura de CSV: {e}")
        raise


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalización de nombres de columnas."""
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace("/", "_")
    )
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputación de valores nulos."""
    cols: List[str] = ["saving_accounts", "checking_account"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de variables para Credit Risk."""
    if "credit_amount" in df.columns and "duration" in df.columns:
        safe_duration = df["duration"].replace(0, 1)
        df["credit_per_month"] = df["credit_amount"] / safe_duration

    if "age" in df.columns and "duration" in df.columns:
        safe_duration = df["duration"].replace(0, 1)
        df["age_duration_ratio"] = df["age"] / safe_duration

    return df


def build_preprocess_pipeline() -> ColumnTransformer:
    """Construcción del ColumnTransformer con caché de joblib."""
    memory: Memory = Memory(location=str(CACHE_DIR), verbose=0)

    num_features: List[str] = [
        "age", "credit_amount", "duration",
        "credit_per_month", "age_duration_ratio"
    ]
    ord_features: List[str] = ["saving_accounts", "checking_account"]
    cat_features: List[str] = ["sex", "housing", "purpose", "job"]

    ord_0: List[str] = ["unknown", "little", "moderate", "quite rich", "rich"]
    ord_1: List[str] = ["unknown", "little", "moderate", "rich"]

    num_pipe: Pipeline = Pipeline(steps=[
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", RobustScaler()),
    ], memory=memory)

    return ColumnTransformer(transformers=[
        ("num", num_pipe, num_features),
        ("ord", OrdinalEncoder(categories=[ord_0, ord_1]), ord_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_features
        ),
    ], remainder="drop")


def prepare_data(
    path: Union[Path, str]
) -> Tuple[pd.DataFrame, pd.Series[int]]:
    """Orquestador completo del preprocesamiento con tipado estricto."""
    df: pd.DataFrame = load_data(path)
    df = add_features(handle_missing(clean_columns(df)))

    if "risk" not in df.columns:
        logger.error("La columna objetivo 'risk' no está en el dataset.")
        raise KeyError("Falta columna 'risk'")

    # Mapeo explícito a enteros para evitar confusiones de tipo
    df["risk"] = df["risk"].map({"good": 0, "bad": 1}).fillna(1).astype(int)

    x_data: pd.DataFrame = df.drop("risk", axis=1)
    # Especificamos que la Serie es de enteros [int]
    y_data: pd.Series[int] = df["risk"]

    return x_data, y_data


def main() -> None:
    """Punto de entrada para ejecución independiente."""
    try:
        logger.info("Ejecutando preprocesamiento independiente...")
        x_data, y_data = prepare_data(INPUT_PATH)

        # Sonar: Usamos '_' para variables no utilizadas
        x_train, _, _, _ = train_test_split(
            x_data, y_data, test_size=0.2, stratify=y_data, random_state=42
        )

        preprocessor = build_preprocess_pipeline()
        preprocessor.fit(x_train)

        # Persistencia de objetos
        out_file = OUTPUT_DIR / "preprocessor.pkl"
        joblib.dump(preprocessor, out_file)
        logger.info(f"Objetos guardados en {OUTPUT_DIR}")

    except Exception as e:
        logger.critical(f"Fallo en el servicio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
