from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, cast

import shap
import pandas as pd
import numpy as np
import numpy.typing as npt
import streamlit as st
from sklearn.pipeline import Pipeline

from .model_service import ModelService

# Configuración de logger para ambiente Docker
logger = logging.getLogger(__name__)


class ShapService:
    """
    Servicio SHAP: Refactorizado para cumplimiento estricto de tipos
    y limpieza de valores para el frontend.
    """

    def __init__(self, model_service: ModelService) -> None:
        """Inicializa explainer garantizando fondo numérico float64."""
        if model_service.model is None:
            raise ValueError("No se puede iniciar SHAP sin un modelo.")

        self.model_service = model_service

        # 1. Resolución de arquitectura
        if isinstance(model_service.model, Pipeline):
            self.pipeline: Pipeline = model_service.model
            # steps[-1][1] extrae el estimador final del pipeline
            self.model_backend: Any = self.pipeline.steps[-1][1]
            self.preprocessor: Optional[Any] = self.pipeline[:-1]
        else:
            self.model_backend = model_service.model
            self.preprocessor = None

        # 2. Carga del Masker (Usamos feature_names del ModelService)
        bg_raw = self._load_bg(
            self.preprocessor,
            self.model_service.feature_names
        )

        # Aseguramos que sea un array de float64 para el explainer
        self.bg_transformed: npt.NDArray[np.float64] = (
            bg_raw.astype(np.float64)
        )

        # 3. Inicialización del explainer usando la función de probabilidad
        self.explainer: shap.Explainer = shap.Explainer(
            self.model_backend.predict_proba,
            masker=self.bg_transformed
        )

    @staticmethod
    @st.cache_data
    def _load_bg(
        preprocessor: Any,
        features: List[str]
    ) -> npt.NDArray[np.float64]:
        """Carga el fondo asegurando compatibilidad con Mypy --strict."""
        try:
            path: str = "data/processed/clean_data.csv"
            bg: pd.DataFrame = pd.read_csv(path).head(100)

            drop_list: List[str] = ["Risk", "risk", "Unnamed: 0"]
            bg = bg.drop(columns=[c for c in drop_list if c in bg.columns])

            if preprocessor is not None:
                transformed = preprocessor.transform(bg)
            else:
                bg_processed = pd.get_dummies(bg)
                for col in features:
                    if col not in bg_processed.columns:
                        bg_processed[col] = 0
                transformed = bg_processed[features].to_numpy()

            # Conversión segura para evitar nulos en el masker
            result = np.nan_to_num(transformed.astype(np.float64))
            return cast(npt.NDArray[np.float64], result)

        except Exception as e:
            logger.error(f"Error en Masker SHAP: {e}")
            # Retorno de emergencia con dimensiones correctas
            return np.zeros((1, len(features)), dtype=np.float64)

    def explain(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula SHAP con limpieza de finitos para el frontend."""
        df: pd.DataFrame = pd.DataFrame([client_data])

        # Usamos feature_names para consistencia con el ModelService
        if self.preprocessor is not None:
            x_trans = self.preprocessor.transform(df)
        else:
            df_dummies = pd.get_dummies(df)
            for col in self.model_service.feature_names:
                if col not in df_dummies.columns:
                    df_dummies[col] = 0
            feat_list = self.model_service.feature_names
            x_trans = df_dummies[feat_list].to_numpy()

        x_numeric = np.nan_to_num(x_trans.astype(np.float64))
        res: Any = self.explainer(x_numeric)

        # Extraemos valores SHAP
        val: npt.NDArray[np.float64] = res.values[0]
        base: Any = res.base_values[0]

        # Ajuste para modelos de clasificación binaria (clase 1)
        if len(val.shape) > 1 and val.shape[-1] == 2:
            val = val[:, 1]
            if isinstance(base, (np.ndarray, list, pd.Series)):
                base = base[1]

        # Limpieza de nulos y tipos para JSON (frontend)
        clean_values: List[float] = [
            float(v) if np.isfinite(v) else 0.0
            for v in val.flatten()
        ]

        return {
            "status": "success",
            "base_value": float(base) if np.isfinite(float(base)) else 0.5,
            "values": clean_values,
            "features": list(client_data.keys()),
            "data": [
                str(d) if isinstance(d, (str, bool)) else float(d)
                for d in df.iloc[0].values
            ],
        }
