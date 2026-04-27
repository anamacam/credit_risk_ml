import sys
from pathlib import Path


def set_model_as_active(run_id):
    """Guarda el Run ID en el archivo de referencia para la API."""
    # Asegurar que la carpeta de artefactos existe
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    active_file = artifacts_dir / "ACTIVE_MODEL.txt"

    with open(active_file, "w", encoding="utf-8") as f:
        f.write(run_id)

    print("\n" + "=" * 50)
    print("✅ MODELO ACTUALIZADO")
    print(f"🆔 Run ID: {run_id}")
    print(f"📄 Archivo: {active_file.absolute()}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        set_model_as_active(sys.argv[1])
    else:
        print("❌ Error: Debes proporcionar un Run ID.")
        # Usamos r"" (raw string) para evitar el error de escape \s
        print(r"Ejemplo: python scripts\set_active_model.py HASH_ID")
