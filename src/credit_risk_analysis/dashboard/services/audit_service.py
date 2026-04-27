from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from credit_risk_analysis.utils.logging_config import setup_logger

# [cite: 2026-03-04] Usamos el logger centralizado para consistencia
logger = setup_logger(name="audit_service")


class AuditService:
    """Servicio profesional para auditoría de decisiones del modelo."""

    def __init__(self) -> None:
        """Configura el logger de auditoría con persistencia dinámica."""
        # [cite: 2026-02-27] Resolución dinámica para Docker/WSL
        root = Path(__file__).resolve().parents[3]
        log_dir = root / "logs"

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"No se pudo crear el directorio de logs: {e}")

        self.audit_path = log_dir / "audit_log.jsonl"
        self._audit_logger = logging.getLogger("RiskAudit")

        # Evitamos duplicidad de handlers en Streamlit (hot-reload)
        if not self._audit_logger.handlers:
            handler = logging.FileHandler(self.audit_path, encoding="utf-8")
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self._audit_logger.addHandler(handler)
            self._audit_logger.setLevel(logging.INFO)

    def log_prediction(
        self,
        input_data: Dict[str, Any],
        probability: float,
        shap_values: Dict[str, Any]
    ) -> None:
        """
        Registra una traza de auditoría en formato JSONL.
        [cite: 2026-03-04] Tipado estricto y redondeo profesional.
        """
        try:
            # [cite: 2026-02-27] Evitamos datos harcodeados o nulos
            features = shap_values.get("features", [])

            audit_entry: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "input": input_data,
                "probability": round(probability, 4),
                "top_shap_features": features[:5] if features else []
            }

            # Registro en el archivo JSONL
            self._audit_logger.info(json.dumps(audit_entry))
            logger.info(f"✅ Auditoría registrada para prob: {probability:.4f}")

        except Exception as e:
            logger.error(f"❌ Fallo al registrar auditoría: {e}")
