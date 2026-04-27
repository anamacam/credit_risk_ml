import pandas as pd
import numpy as np
import mlflow.pyfunc
from typing import Any


class ModelInference:
    def __init__(self, model_name: str, stage: str = "Production"):
        self.model_name = model_name
        self.stage = stage
        self.model: Any = None
        self.preprocessor: Any = None

    def load_production_artifacts(self) -> None:
        """
        Descarga el modelo y los artefactos necesarios desde MLflow.
        """
        model_uri = f"models:/{self.model_name}/{self.stage}"
        # Cargamos el modelo como un pyfunc para máxima compatibilidad
        self.model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Artefactos cargados desde: {model_uri}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones previas a la inferencia.
        """
        return df

    def predict(self, data: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """
        Realiza la inferencia sobre los datos de entrada.
        """
        if self.model is None:
            raise RuntimeError(
                "El modelo no ha sido cargado. "
                "Llama a load_production_artifacts() primero."
            )

        processed_data = self.preprocess_data(data)
        predictions = self.model.predict(processed_data)

        return np.array(predictions)


# --- Ejemplo de ejecución ---
if __name__ == "__main__":
    inference_engine = ModelInference(model_name="credit_risk_model")

    try:
        inference_engine.load_production_artifacts()

        # Datos de prueba
        raw_data = pd.DataFrame({
            "feature1": [1.0, 2.5],
            "feature2": [0, 1]
        })

        results = inference_engine.predict(raw_data)
        print(f"Resultados de inferencia: {results}")

    except Exception as e:
        print(f"❌ Error en el proceso: {e}")
