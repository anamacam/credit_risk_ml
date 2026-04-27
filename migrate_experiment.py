import os
from pathlib import Path
import mlflow

# 1. CONFIGURACIÓN (Rutas Absolutas)
RUN_ID = "f6ec06d592954560946dd1365a01d6dc"
BASE_DIR = Path(__file__).resolve().parent
# Apuntamos directo a los artefactos confirmados
LOCAL_ART_DIR = BASE_DIR / "mlruns" / "1" / RUN_ID / "artifacts"
REMOTE_URI = "http://localhost:5000"


def migrate() -> None:
    """Migra solo los artefactos al servidor remoto."""
    if not LOCAL_ART_DIR.exists():
        print(f"❌ Error: No se encontraron artefactos en {LOCAL_ART_DIR}")
        return

    print(f"📦 Localizados artefactos en: {LOCAL_ART_DIR}")

    # Configurar MLflow remoto
    os.environ["MLFLOW_TRACKING_URI"] = REMOTE_URI
    mlflow.set_tracking_uri(REMOTE_URI)
    mlflow.set_experiment("Credit_Risk_Local_Migration")

    # 2. CREAR RUN Y SUBIR
    with mlflow.start_run(run_name="Threshold_0_2_v1.0_migrated") as run:
        new_id = run.info.run_id

        # Loggear info mínima para trazabilidad
        mlflow.log_param("original_run_id", RUN_ID)

        # Subir el contenido de la carpeta 'artifacts'
        mlflow.log_artifacts(str(LOCAL_ART_DIR))

        # Fix F541/S3457: Quitamos la 'f' si no hay llaves {}
        print("🎉 ¡ÉXITO! Artefactos migrados al servidor remoto.")
        print(f"🆔 Nuevo Run ID: {new_id}")

        # E501: URL fragmentada para cumplir con PEP 8
        exp_id = run.info.experiment_id
        print(f"🌐 UI: {REMOTE_URI}/#/experiments/{exp_id}/runs/{new_id}")


if __name__ == "__main__":
    migrate()
