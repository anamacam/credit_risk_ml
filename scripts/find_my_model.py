import mlflow
from pathlib import Path


def main():
    """Localiza la ruta física del modelo activo en el MLflow Store."""
    # 1. Leer el ID que activamos
    active_path = Path("artifacts/ACTIVE_MODEL.txt")

    if not active_path.exists():
        print(f"❌ Error: No existe el archivo {active_path}")
        return

    with open(active_path, "r", encoding="utf-8") as f:
        run_id = f.read().strip()

    try:
        # 2. MLflow localiza la ruta real del artefacto
        print(f"🔎 Buscando ruta física para el ID: {run_id}...")
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model"
        )

        print("\n" + "=" * 60)
        print("✅ ¡ENCONTRADO!")
        print(f"📍 RUTA FÍSICA: {local_path}")
        print("📂 Contenido de la carpeta:")
        for item in Path(local_path).iterdir():
            print(f"  - {item.name}")
        print("=" * 60 + "\n")

    except Exception:
        print(f"\n❌ Error: No se encuentra el ID {run_id} en mlruns.")
        print("Verifica que el ID en ACTIVE_MODEL.txt sea correcto.")


if __name__ == "__main__":
    main()
