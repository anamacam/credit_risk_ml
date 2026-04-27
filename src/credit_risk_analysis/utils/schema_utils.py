from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any


def get_api_example() -> Dict[str, Any]:
    """Genera dinámicamente el ejem de la API desde YAML [cite: 2026-03-04]."""
    # Localizamos el archivo subiendo desde 'utils' a 'src' y luego a 'config'
    base_path = Path(__file__).parent.parent
    schema_path = base_path / "config" / "schema.yaml"

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            # Retornamos solo nombre: default (lo que la API necesita)
            return {
                feat["name"]: feat.get("default")
                for feat in config.get("features", [])
            }
    except Exception as e:
        # Fallback para Mypy y Docker en caso de error de lectura
        print(f"Error cargando schema.yaml: {e}")
        return {"Age": 30, "Credit amount": 5000}
