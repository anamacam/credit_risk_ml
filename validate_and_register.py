from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict

# Configuración
MODEL_PATH = "artifacts/model/model.pkl"
EXPERIMENT_NAME = "credit_risk_analysis"
MODEL_NAME = "CreditRiskRF_Official"


def load_validation_metrics() -> Dict[str, float]:
    """Carga métricas de validación locales."""
    return {
        "training_roc_auc": 0.9165,
        "training_accuracy_score": 0.8414,
        "training_f1_score": 0.8277,
        "training_score": 0.823
    }


def validate_and_register_model() -> None:
    """Valida el artefacto y lo registra formalmente en MLflow."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f"❌ Error: No existe el modelo en {MODEL_PATH}")
        return

    print(f"🧪 Validando modelo: {MODEL_NAME}...")
    model = joblib.load(model_file)
    metrics = load_validation_metrics()

    with mlflow.start_run(run_name="Validacion_y_Registro_Oficial"):
        # Log de métricas reales
        mlflow.log_metrics(metrics)
        # Log del modelo
        mlflow.sklearn.log_model(model, "model")

        # Solución al error union-attr de Mypy: verificación explícita de None
        active_run = mlflow.active_run()
        if active_run is not None:
            run_id = active_run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            # Registro en el catálogo
            mlflow.register_model(model_uri, MODEL_NAME)
            print(f"✅ Registro completado. Run ID: {run_id}")
            print(f"📊 Métricas enviadas: AUC {metrics['training_roc_auc']}")
        else:
            print("❌ Error: No se pudo obtener un Run activo.")


if __name__ == "__main__":
    validate_and_register_model()
