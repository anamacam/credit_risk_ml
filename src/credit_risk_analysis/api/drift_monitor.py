"""Calcula el PSI para detectar Data Drift en el modelo de crédito."""

import os
import sys
from typing import Dict, Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from prometheus_client import Gauge

# Métrica para Grafana
DRIFT_PSI = Gauge(
    'credit_model_psi_score',
    'Population Stability Index por feature',
    ['feature_name']
)


def calculate_psi(
    expected: pd.Series[Any],
    actual: pd.Series[Any],
    buckets: int = 10
) -> float:
    """Calcula el PSI entre la distribución esperada y la actual."""

    def scale_range(
        series: pd.Series[Any],
        n_buckets: int
    ) -> npt.NDArray[np.float64]:
        """Retorna los bins como ndarray con tipado estricto."""
        # Se extraen los bordes de los bins usando pandas.cut
        # Multilínea para cumplir con PEP8 (Max 79 chars)
        _, bins = pd.cut(
            series,
            bins=n_buckets,
            retbins=True,
            duplicates='drop'
        )
        return np.array(bins, dtype=np.float64)

    # Obtenemos los puntos de corte basados en la distribución esperada
    breakpoints = scale_range(expected, buckets)

    # Cálculo de histogramas (frecuencias absolutas)
    expected_hist, _ = np.histogram(expected, bins=breakpoints)
    actual_hist, _ = np.histogram(actual, bins=breakpoints)

    # Conversión a porcentajes (frecuencias relativas)
    expected_percents = expected_hist / len(expected)
    actual_percents = actual_hist / len(actual)

    # Evitar división por cero o log(0) para estabilidad numérica
    expected_percents = np.clip(expected_percents, 0.0001, 1.0)
    actual_percents = np.clip(actual_percents, 0.0001, 1.0)

    # Fórmula del PSI: (Actual% - Expected%) * ln(Actual% / Expected%)
    diff = actual_percents - expected_percents
    ratio = actual_percents / expected_percents
    psi_values = diff * np.log(ratio)

    return float(np.sum(psi_values))


def run_drift_check(current_data_df: pd.DataFrame) -> Dict[str, float]:
    """Compara datos actuales contra el baseline .pkl."""
    path_pkl = "/app/artifacts/baseline_stats.pkl"

    if not os.path.exists(path_pkl):
        # Fallback para desarrollo local
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )
        path_pkl = os.path.join(base_path, "artifacts", "baseline_stats.pkl")

    try:
        baseline_stats = joblib.load(path_pkl)
    except Exception as e:
        print(f"ERROR: No se pudo cargar baseline en {path_pkl}: {e}")
        return {}

    psi_results: Dict[str, float] = {}

    if not current_data_df.empty:
        print("\n" + "=" * 40)
        print("--- ANALISIS DE DRIFT (PSI) ---")

        for feature, baseline_series in baseline_stats.items():
            if feature in current_data_df.columns:
                psi_score = calculate_psi(
                    baseline_series,
                    current_data_df[feature]
                )
                psi_results[feature] = psi_score
                DRIFT_PSI.labels(feature_name=feature).set(psi_score)

                if psi_score <= 0.1:
                    status = "OK - ESTABLE"
                elif psi_score <= 0.2:
                    status = "ADVERTENCIA - CAMBIO LEVE"
                else:
                    status = "CRITICO - DRIFT DETECTADO"

                print(
                    f"Feature: {feature:12} | "
                    f"PSI: {psi_score:.4f} | "
                    f"STATUS: {status}"
                )

        print("=" * 40 + "\n")

    sys.stdout.flush()
    return psi_results


if __name__ == "__main__":
    print("Revisando Drift con datos sintéticos...")
    rng = np.random.default_rng(seed=42)

    normal_data = pd.DataFrame({
        "income": rng.normal(45000, 5000, 500),
        "age": rng.normal(35, 5, 500),
        "debt": rng.normal(5000, 1000, 500),
        "housing": rng.choice([0, 1], 500),
        "loan_amount": rng.normal(20000, 4000, 500)
    })

    run_drift_check(normal_data)
