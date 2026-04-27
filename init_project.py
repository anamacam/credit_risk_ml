import shutil
from pathlib import Path


def clean_project() -> None:
    """Inicia la limpieza profesional del proyecto."""
    root = Path.cwd()
    print(f"🧹 Iniciando limpieza profesional en: {root}")

    # 1. Definir rutas clave
    models_dir = root / "src" / "models"
    modeling_dir = root / "src" / "modeling"

    # Crear modeling si no existe
    modeling_dir.mkdir(parents=True, exist_ok=True)

    # 2. Rescatar lógica única de 'models' antes de borrarla
    to_move = ["pipeline.py", "model_factory.py"]
    for file_name in to_move:
        src_file = models_dir / file_name
        dest_file = modeling_dir / file_name
        if src_file.exists():
            print(f"📦 Moviendo {file_name} a src/modeling/...")
            shutil.move(str(src_file), str(dest_file))

    # 3. Eliminar la carpeta 'models' (duplicada/obsoleta)
    if models_dir.exists():
        print("🗑️ Eliminando carpeta obsoleta: src/models")
        shutil.rmtree(models_dir)

    # 4. Limpieza de residuos (Automation)
    patterns_to_remove = [
        "**/__pycache__",
        "**/*.egg-info",
        "**/.pytest_cache",
        "**/.mypy_cache"
    ]

    for pattern in patterns_to_remove:
        for path in root.glob(pattern):
            print(f"🔥 Borrando residuo: {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    print("\n✅ Proyecto organizado. Estructura recomendada aplicada.")
    print("🚀 Ahora usa 'src/training/train_experiment.py' para tus corridas.")


if __name__ == "__main__":
    clean_project()
