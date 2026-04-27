"""Crea baseline estadístico para drift detection del modelo de crédito."""

from typing import Dict, Union
import os
import joblib
import numpy as np
import pandas as pd

# Features del modelo
FEATURES_REALES = [
    "income",
    "age",
    "debt",
    "housing",
    "loan_amount"
]


def create_baseline_from_training_data(
    input_data: Union[str, pd.DataFrame] = "data/training_data.csv"
) -> Dict[str, pd.Series]:
    """Crea baseline stats con datos reales o un DataFrame."""
    # Manejar si es ruta (str) o ya es un DataFrame
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"No se encontró el archivo: {input_data}")
        df = pd.read_csv(input_data)
    else:
        df = input_data

    df_baseline = df[FEATURES_REALES].dropna()
    baseline_stats: Dict[str, pd.Series] = {}

    for feature in FEATURES_REALES:
        # Guardamos la serie para cálculos de PSI posteriores
        baseline_stats[feature] = df_baseline[feature]

        print(f"✅ Baseline {feature}:")
        print(f"   Media: {df_baseline[feature].mean():.1f}")
        print(f"   Std:   {df_baseline[feature].std():.1f}")
        print(
            f"   Rango: "
            f"[{df_baseline[feature].min():.0f}, "
            f"{df_baseline[feature].max():.0f}]"
        )

    # Asegurar que el directorio existe
    os.makedirs("artifacts", exist_ok=True)

    # Guardar para drift_monitor.py
    joblib.dump(baseline_stats, "artifacts/baseline_stats.pkl")
    print("\n🎯 Baseline guardado en artifacts/baseline_stats.pkl")

    return baseline_stats


def main() -> None:
    """Genera datos representativos si no hay CSV real."""
    rng = np.random.default_rng(seed=42)

    # Datos típicos de crédito
    sample_training_data = pd.DataFrame({
        "income": rng.normal(55000, 25000, 10000),
        "age": rng.normal(38, 12, 10000),
        "debt": rng.exponential(15000, 10000),
        "housing": rng.choice([0, 1, 2], 10000, p=[0.4, 0.4, 0.2]),
        "loan_amount": rng.normal(220000, 100000, 10000)
    }).clip(lower=0)

    create_baseline_from_training_data(sample_training_data)


if __name__ == "__main__":
    main()
