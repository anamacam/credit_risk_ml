import mlflow
from sklearn.linear_model import LogisticRegression
from src.models.mlflow_utils import setup_mlflow, log_model_artifact


def train_and_log_dummy_model() -> None:
    """Entrena un modelo rápido y lo registra en MLflow."""
    setup_mlflow()

    with mlflow.start_run(run_name="Model_Registration_Test"):
        # 1. Creamos un modelo de prueba
        model = LogisticRegression()
        # (Aquí iría tu fit con datos reales)

        # 2. Registramos parámetros y métricas
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_metric("accuracy", 0.85)

        # 3. GUARDADO AUTOMÁTICO DEL MODELO
        log_model_artifact(model)

        print("✅ Modelo registrado exitosamente en MLflow.")


if __name__ == "__main__":
    train_and_log_dummy_model()
