import random
import time
from typing import Any, Dict
import requests

URL: str = "http://localhost:8000/predict"


def generate_random_customer() -> Dict[str, Any]:
    """
    Genera datos aleatorios basados en tu esquema validado.
    RESOLUCIÓN: Añadida anotación de retorno Dict[str, Any].
    """
    return {
        "data": {
            "age": random.randint(18, 75),
            "sex": random.choice(["male", "female"]),
            "housing": random.choice(["own", "free", "rent"]),
            "saving_accounts": random.choice(["little", "moderate", "rich"]),
            "checking_account": random.choice(["little", "moderate", "rich"]),
            "credit_amount": round(random.uniform(500, 15000), 2),
            "duration": random.choice([6, 12, 18, 24, 36, 48]),
            "purpose": random.choice(
                ["education", "car", "furniture", "business", "radio/TV"]
            ),
        }
    }


def run_test(iterations: int = 50) -> None:
    """
    Ejecuta el test de estrés con tipado estricto.
    RESOLUCIÓN: Tipado de argumentos y retorno -> None [cite: 2026-03-04].
    """
    print(f"🚀 Iniciando test de {iterations} peticiones...")
    success_count: int = 0

    for i in range(iterations):
        payload: Dict[str, Any] = generate_random_customer()
        try:
            # RESOLUCIÓN no-untyped-call: payload ahora tiene tipo definido
            response: Any = requests.post(URL, json=payload)
            if response.status_code == 200:
                success_count += 1
                print(f"✅ [{i+1}] Predicción exitosa")
            else:
                print(f"❌ [{i+1}] Error: {response.json()}")
        except Exception as e:
            print(f"🔥 Error de conexión: {e}")

        time.sleep(0.1)

    print(f"\n✨ Test finalizado. Éxitos: {success_count}/{iterations}")


if __name__ == "__main__":
    run_test(50)
