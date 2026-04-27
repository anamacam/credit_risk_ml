import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_config_path() -> Path:
    """Retorna la ruta del archivo de configuración."""
    return Path(__file__).parent / "schema.yaml"


def load_config() -> Dict[str, Any]:
    """Carga YAML y sobreescribe con variables de entorno (.env)."""
    path = get_config_path()

    # 1. Cargar base desde YAML
    with open(path, "r", encoding="utf-8") as f:
        config = dict(yaml.safe_load(f))

    # 2. Sobreescritura dinámica (Prioridad MLOps)
    # Si existe en el entorno/OS, usamos eso (ideal para Docker)
    config["mlflow_tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI", config.get("mlflow_tracking_uri")
    )
    config["model_path"] = os.getenv(
        "MODEL_PATH", config.get("model_path")
    )
    config["log_level"] = os.getenv(
        "LOG_LEVEL", config.get("log_level", "INFO")
    )

    return config


# Instancia lista para usar
config = load_config()
