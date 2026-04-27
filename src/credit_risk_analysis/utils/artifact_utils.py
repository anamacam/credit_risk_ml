from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_artifact_integrity(path: Path) -> bool:
    """
    Valida la existencia y el tamaño de los artefactos.
    [cite: 2026-03-07] Evita cargar archivos vacíos o corruptos.
    """
    try:
        if not path.exists():
            logger.warning(f"⚠️ Artefacto no encontrado en: {path}")
            return False

        # Validación de integridad mínima: que no sea un archivo de 0 bytes
        if path.is_file() and path.stat().st_size == 0:
            logger.error(f"🚨 Artefacto corrupto (0 bytes): {path}")
            return False

        return True
    except Exception as e:
        logger.error(f"❌ Error verificando integridad: {e}")
        return False


def get_valid_model_path(base_dir: str = "artifacts") -> Path:
    """
    Resuelve la ruta absoluta del modelo para evitar fallos de contexto.
    [cite: 2026-03-07] Soporta rutas relativas desde la raíz del proyecto.
    """
    # Buscamos la raíz del proyecto de forma dinámica
    project_root = Path(__file__).resolve().parents[3]

    # Construimos la ruta hacia el modelo (versión simplificada para local)
    model_path = project_root / base_dir / "model" / "model.pkl"

    return model_path
