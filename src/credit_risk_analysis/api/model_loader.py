from __future__ import annotations

import os
import pickle
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Configuración de logs
logger = logging.getLogger(__name__)

# Definición de rutas basadas en la estructura del contenedor
DEFAULT_MODEL_PATH = "/app/artifacts/model/model.pkl"
DEFAULT_PREP_PATH = "/app/artifacts/temp/preprocessor.pkl"

MODEL_FILE_PATH: Path = Path(
    os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
)
PREP_FILE_PATH: Path = Path(
    os.getenv("PREP_PATH", DEFAULT_PREP_PATH)
)


class ModelService:
    """
    Servicio profesional de inferencia para Riesgo Crediticio.
    Gestiona la carga de artefactos y la lógica de decisión dinámica.
    """

    def __init__(self) -> None:
        # Inicializamos los atributos para evitar AttributeError
        self.model: Any | None = None
        self.preprocessor: Any | None = None
        self.threshold: float = float(os.getenv("MODEL_THRESHOLD", 0.30))
        self.is_initialized: bool = False

    def initialize(self) -> None:
        """
        Carga el modelo y el preprocesador desde el volumen de artefactos.
        """
        try:
            # 1. Carga del modelo
            if not MODEL_FILE_PATH.exists():
                msg = f"No existe el modelo en: {MODEL_FILE_PATH}"
                raise FileNotFoundError(msg)

            with open(MODEL_FILE_PATH, "rb") as f:
                self.model = pickle.load(f)

            # 2. Carga del preprocesador (Aquí va el bloque que preguntaste)
            if PREP_FILE_PATH.exists():
                with open(PREP_FILE_PATH, "rb") as f:
                    self.preprocessor = pickle.load(f)
                logger.info("✅ Preprocesador cargado correctamente.")
            else:
                logger.warning(
                    "⚠️ No se encontró preprocesador en %s", PREP_FILE_PATH
                )

            self.is_initialized = True
            logger.info(
                "🚀 ModelService inicializado con umbral: %s", self.threshold
            )

        except Exception as exc:
            logger.error("❌ Fallo en la inicialización de ML: %s", str(exc))
            self.is_initialized = False
            raise RuntimeError("Servicios de ML no disponibles") from exc

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Realiza la predicción dinámica: Transforma -> Infiere -> Decide.
        """
        if not self.is_initialized or self.model is None:
            self.initialize()

        df_input = pd.DataFrame([data])

        # Uso del preprocesador cargado
        if self.preprocessor:
            try:
                x_input = self.preprocessor.transform(df_input)
            except Exception as e:
                logger.error("Error en transformación de datos: %s", e)
                raise ValueError("Formato de entrada inválido") from e
        else:
            x_input = df_input

        # Inferencia
        if self.model is not None:
            probabilities = self.model.predict_proba(x_input)[0]
            risk_score = float(probabilities[1])
        else:
            raise RuntimeError("El modelo no se cargó correctamente.")

        decision = "high_risk" if risk_score >= self.threshold else "low_risk"

        return {
            "probability": round(risk_score, 4),
            "decision": decision,
            "threshold": self.threshold,
            "status": "success"
        }
