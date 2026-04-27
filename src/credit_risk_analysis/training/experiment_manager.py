from __future__ import annotations

import json
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Configuración de logger
logger = logging.getLogger(__name__)


class ExperimentManager:
    """Gestiona el almacenamiento de artefactos de experimentos de ML."""

    def __init__(self, experiment_name: str) -> None:
        """Inicializa directorios con timestamp único."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name: str = f"{timestamp}_{experiment_name}"
        self.base_path: Path = Path("experiments") / self.exp_name
        self.plots_path: Path = self.base_path / "plots"

        # Creación robusta de directorios
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Guarda la configuración del modelo en formato JSON."""
        config_path = self.base_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """Guarda las métricas de evaluación en formato JSON."""
        metrics_path = self.base_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    def save_model(self, model: Any) -> None:
        """Serializa el modelo usando joblib con compresión."""
        model_path = self.base_path / "model.pkl"
        joblib.dump(model, model_path, compress=3)

    def save_roc_curve(
        self,
        y_true: Union[pd.Series[Any], npt.NDArray[np.int64]],
        y_proba: npt.NDArray[np.float64]
    ) -> None:
        """Genera y guarda la curva ROC."""
        # Se asegura compatibilidad de tipos para sklearn y Mypy
        RocCurveDisplay.from_predictions(
            cast(Any, y_true),
            cast(Any, y_proba)
        )
        plt.title(f"ROC Curve - {self.exp_name}")
        plt.savefig(self.plots_path / "roc_curve.png")
        plt.close()

    def save_pr_curve(
        self,
        y_true: Union[pd.Series[Any], npt.NDArray[np.int64]],
        y_proba: npt.NDArray[np.float64]
    ) -> None:
        """Genera y guarda la curva Precision-Recall."""
        PrecisionRecallDisplay.from_predictions(
            cast(Any, y_true),
            cast(Any, y_proba)
        )
        plt.title(f"Precision-Recall Curve - {self.exp_name}")
        plt.savefig(self.plots_path / "pr_curve.png")
        plt.close()
