import mlflow
import pandas as pd
from pathlib import Path

# Configuración de MLflow
MLFLOW_DB_PATH = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_DB_PATH)

EXPERIMENT_ID = "2"
MODEL_FOLDER = "m-20b9ea00c84a415d90a5cbf819fa89f6"
base_path = Path("C:/projects/credit_risk_ml/mlruns")
model_uri = str(
    base_path / EXPERIMENT_ID / "models" / MODEL_FOLDER / "artifacts"
)


def run_test() -> None:
    """Prueba de inferencia con el esquema exacto que pide el modelo."""
    print(f"🔍 Cargando desde: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Modelo cargado correctamente.")

        # El esquema exige variables Dummy/One-Hot (booleanas)
        # Se añaden todas las columnas requeridas por el log de error
        data = {
            "Age": 90,
            "Job": 0,
            "Credit amount": 1000000,
            "Duration": 72,
            "Sex_male": True,
            "Housing_own": False,
            "Housing_rent": False,
            "Saving accounts_moderate": False,
            "Saving accounts_quite rich": False,
            "Saving accounts_rich": False,
            "Checking account_moderate": False,
            "Checking account_rich": False,
            "Purpose_car": False,
            "Purpose_domestic appliances": False,
            "Purpose_education": False,
            "Purpose_furniture/equipment": False,
            "Purpose_radio/TV": False,
            "Purpose_repairs": False,
            "Purpose_vacation/others": True
        }

        df = pd.DataFrame([data])
        prediction = model.predict(df)

        print("\n--- RESULTADO DEL MODELO ---")
        print(f"Salida: {prediction}")

        is_multi = hasattr(prediction, "shape") and len(prediction.shape) > 1
        if is_multi:
            print(f"Probabilidad Riesgo: {float(prediction[0][1]):.4f}")
        else:
            print(f"Valor: {float(prediction[0])}")

    except Exception as e:
        print(f"❌ Error durante la inferencia: {e}")


if __name__ == "__main__":
    run_test()
