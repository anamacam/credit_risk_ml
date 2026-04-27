import os
import pandas as pd


def validate_raw_data(file_path: str) -> bool:
    """
    Valida que el CSV tenga las columnas necesarias para el riesgo.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: El archivo {file_path} no existe.")
        return False

    try:
        # Leemos solo la cabecera para ser eficientes
        df = pd.read_csv(file_path, nrows=1)

        # Definimos las columnas mínimas viables
        required_cols = ["Age", "Credit amount", "Duration"]

        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print(f"❌ Error: Faltan columnas críticas: {missing}")
            print(f"Columnas detectadas: {list(df.columns)}")
            return False

        print("✅ Validación de datos exitosa.")
        return True

    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
        return False
