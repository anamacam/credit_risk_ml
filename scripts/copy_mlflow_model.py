"""
Copia el modelo de MLflow a una ubicación permanente en el proyecto.
"""

import os
import shutil
from pathlib import Path
import mlflow
from dotenv import load_dotenv

load_dotenv()


def main():
    """Función principal para copiar el modelo activo."""
    print("=" * 80)
    print("📦 COPIANDO MODELO DE MLFLOW")
    print("=" * 80)

    # Configurar MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Obtener Run ID
    run_id = os.getenv("RUN_ID")
    if not run_id:
        active_file = Path("artifacts/ACTIVE_MODEL.txt")
        with open(active_file, 'r', encoding="utf-8") as f:
            run_id = f.read().strip()

    print(f"\n🆔 Run ID: {run_id}")

    # Descargar modelo
    print("\n📥 Descargando modelo de MLflow...")

    model_uri = f"runs:/{run_id}/model"
    temp_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    print(f"    ✅ Descargado en: {temp_path}")

    # Crear directorio permanente
    permanent_dir = Path("artifacts/active_model")
    permanent_dir.mkdir(parents=True, exist_ok=True)

    # Copiar archivos
    print("\n📋 Copiando archivos a ubicación permanente...")

    temp_dir = Path(temp_path)

    files_to_copy = [
        "model.pkl",
        "preprocessor.pkl",
        "MLmodel",
        "conda.yaml",
        "requirements.txt"
    ]

    copied = []
    for file_name in files_to_copy:
        src = temp_dir / file_name
        if src.exists():
            dst = permanent_dir / file_name
            shutil.copy2(src, dst)
            copied.append(file_name)
            print(f"    ✅ {file_name}")
        else:
            print(f"    ⚠️ {file_name} no encontrado (opcional)")

    # Actualizar .env
    print("\n📝 Actualizando .env...")

    env_content = f"""# Modelo Activo
RUN_ID={run_id}
MODEL_PATH=artifacts/active_model/model.pkl
PREPROCESSOR_PATH=artifacts/active_model/preprocessor.pkl
MLFLOW_TRACKING_URI=http://localhost:5000
DEFAULT_THRESHOLD=0.387

# API Configuration
API_VERSION=1.4.0
"""

    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)

    print("    ✅ .env actualizado")

    # Verificar
    print("\n🔍 Verificando archivos copiados...")

    model_file = permanent_dir / "model.pkl"
    preprocessor_file = permanent_dir / "preprocessor.pkl"

    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"    ✅ model.pkl ({size_mb:.2f} MB)")
    else:
        print("    ❌ model.pkl NO encontrado")

    if preprocessor_file.exists():
        size_kb = preprocessor_file.stat().st_size / 1024
        print(f"    ✅ preprocessor.pkl ({size_kb:.2f} KB)")
    else:
        print("    ⚠️ preprocessor.pkl NO encontrado (opcional)")

    print("\n" + "=" * 80)
    print("✅ MODELO COPIADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\n📁 Ubicación: {permanent_dir.absolute()}")
    print(f"📄 Archivos: {', '.join(copied)}")
    print("\n💡 Siguiente paso:")
    print("1. Reinicia la API")
    print(f"2. Verifica que cargue desde: {permanent_dir}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
