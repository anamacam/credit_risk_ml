# src/utils/io_utils.py
from pathlib import Path
from typing import Union


def check_artifact_integrity(path: str = "artifacts/model/model.pkl") -> bool:
    """
    Verifica si el artefacto existe y no está vacío [cite: 2026-03-04].
    Ayuda a diagnosticar errores de volúmenes en Docker [cite: 2026-02-27].
    """
    file_path: Path = Path(path)

    if not file_path.exists():
        print(f"⚠️ Error: La ruta {path} no existe.")
        return False

    # Verificamos que no sea un archivo de 0 bytes
    if file_path.stat().st_size == 0:
        print(f"⚠️ Error: El archivo en {path} está vacío (0 bytes).")
        return False

    return True


def get_valid_model_path(search_dir: str = "artifacts") -> Union[Path, str]:
    """
    Busca el modelo de forma recursiva para evitar errores [cite: 2026-03-04].
    Ignora rutas corruptas generadas por Docker [cite: 2026-02-27].
    """
    # Buscamos en la carpeta de entrada (por defecto artifacts)
    for path in Path(search_dir).rglob("model.pkl"):
        # Ignoramos rutas corruptas con caracteres raros (C)
        if "C:" not in str(path):
            return path

    # Si no lo encuentra en artifacts, busca en mlruns como backup
    for path in Path("mlruns").rglob("model.pkl"):
        if "C:" not in str(path):
            return path

    raise FileNotFoundError("No encontró el model en artifacts/ ni en mlruns/")
