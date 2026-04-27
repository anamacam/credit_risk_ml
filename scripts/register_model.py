import pickle
import mlflow.pyfunc
import pandas as pd
from typing import Dict, Any
from sklearn.pipeline import Pipeline

# F401: Importado para que MLflow registre la dependencia del pipeline
from services.utils import CreditFeatureEngineer  # noqa: F401


class CreditRiskWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper para empaquetar lógica personalizada en MLflow."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Carga los artefactos (el pipeline original)."""
        with open(context.artifacts["pipeline"], "rb") as f:
            self.pipeline: Pipeline = pickle.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame
    ) -> Any:
        """Predicción estandarizada para el dashboard."""
        return self.pipeline.predict_proba(model_input)


def save_and_register_model(pipeline: Pipeline, model_name: str) -> None:
    """Guarda y registra el modelo de forma profesional."""
    temp_path: str = "models/temp_pipeline.pkl"
    with open(temp_path, "wb") as f:
        pickle.dump(pipeline, f)

    artifacts: Dict[str, str] = {"pipeline": temp_path}

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=CreditRiskWrapper(),
            artifacts=artifacts,
            code_path=["src/dashboard/services/utils.py"]
        )
