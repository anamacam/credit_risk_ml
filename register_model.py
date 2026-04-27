from __future__ import annotations

import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

# --- CONFIGURACIÓN ---
MODEL_PATH = "artifacts/model/model.pkl"
EXPERIMENT_NAME = "Credit_Risk_Analysis"
MODEL_NAME = "CreditRiskRF_Official"


def register_model_mlflow() -> None:
    """Registra el modelo entrenado en el Model Registry de MLflow."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: No se encontró el archivo en {MODEL_PATH}")
        return

    print(f"📦 Cargando modelo desde {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    with mlflow.start_run(run_name="Registro_Manual_Oficial"):
        # Log del modelo
        mlflow.sklearn.log_model(model, "model")

        # Solución al error union-attr: validamos que exista un run activo
        active_run = mlflow.active_run()
        if active_run is not None:
            run_id = active_run.info.run_id
            model_uri = f"runs:/{run_id}/model"

            # Registro en el Model Registry
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"✅ Modelo registrado con éxito. Run ID: {run_id}")
        else:
            print("❌ Error: No se pudo obtener un Run activo de MLflow.")


if __name__ == "__main__":
    register_model_mlflow()
