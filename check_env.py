import os
from pathlib import Path
from dotenv import load_dotenv


def verify_setup() -> None:
    """Verifica la integridad de las variables de entorno."""
    env_path = Path(".env")
    print(f"--- Verificando archivo .env en: {env_path.absolute()} ---")

    if not env_path.exists():
        print("❌ ERROR: No se encuentra el archivo .env")
        return

    load_dotenv()

    threshold = os.getenv("DEFAULT_THRESHOLD")
    model_p = os.getenv("MODEL_PATH")

    print(f"✅ DEFAULT_THRESHOLD: {threshold}")
    print(f"✅ MODEL_PATH: {model_p}")

    if threshold == "0.387":
        print("\n¡PERFECTO! La sincronizacion es correcta.")
    else:
        msg = "\n⚠️ ALERTA: El valor detectado no es 0.387."
        print(msg)


if __name__ == "__main__":
    verify_setup()
