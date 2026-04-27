import sqlite3
from typing import Any
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de ruta con tipado explícito
DB_PATH: Path = Path("logs/predictions.db")


def generate_report() -> None:
    """
    Genera visualizaciones basadas en los logs de la base de datos.
    Resuelve deuda técnica E501 y W291 [cite: 2026-03-04].
    """
    if not DB_PATH.exists():
        print("Error: No se encontró la base de datos.")
        return

    # 1. Cargar datos (RESOLUCIÓN E501: Línea dividida para < 79 chars)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query: str = "SELECT * FROM model_logs"
            df: pd.DataFrame = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error al leer la base de datos: {e}")
        return

    if df.empty:
        print("La base de datos está vacía. Realiza predicciones primero.")
        return

    # Convertir timestamp a objeto datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Configurar la figura
    _, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3)

    # GRAFICA 1: Distribución de Probabilidades
    sns.histplot(
        df["probability"], bins=10, kde=True, ax=axes[0], color="skyblue"
    )
    axes[0].axvline(0.27, color="red", linestyle="--", label="Umbral (0.27)")
    axes[0].set_title("Distribución de Probabilidades de Riesgo")
    axes[0].set_xlabel("Probabilidad (Clase Bad)")
    axes[0].legend()

    # GRAFICA 2: Principales Factores de Riesgo (SHAP)
    # RESOLUCIÓN W291: Limpieza de espacios en blanco en esta sección
    order_idx: Any = df["top_feature"].value_counts().index
    sns.countplot(
        y="top_feature",
        data=df,
        ax=axes[1],
        palette="viridis",
        order=order_idx,
    )
    axes[1].set_title("Factores más Influyentes (SHAP)")
    axes[1].set_xlabel("Frecuencia de Aparición")
    axes[1].set_ylabel("Variable")

    print(f"Reporte generado con éxito basado en {len(df)} registros.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_report()
