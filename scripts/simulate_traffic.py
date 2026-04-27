import random
import time

import requests

# Configuración
URL = "http://127.0.0.1:8000/predict"


def simulate_traffic() -> None:
    """Envia datos de prueba para activar las métricas en Grafana."""
    print("🚀 Enviando datos de prueba para activar Grafana...")
    for i in range(50):
        payload = {
            "Age": random.randint(18, 80),
            "Credit amount": random.randint(500, 20000),
            "Duration": random.randint(6, 60),
            "Job": random.randint(0, 3),
            "Sex": random.choice(["male", "female"]),
            "Housing": random.choice(["own", "rent", "free"]),
            "Saving accounts": random.choice(["little", "moderate", "rich"]),
            "Checking account": random.choice(["little", "moderate", "rich"]),
            "Purpose": random.choice(["car", "business", "furniture"])
        }
        try:
            response = requests.post(URL, json=payload, timeout=5)
            data = response.json()
            prob = data.get('probability', 0.0)
            dec = data.get('decision', 'N/A')
            print(f"[{i+1}] Prob: {prob:.2f} | {dec}")
        except Exception as e:
            print(f"❌ Error en petición {i+1}: {e}")

        time.sleep(1)  # Dos espacios antes del comentario inline


if __name__ == "__main__":
    simulate_traffic()
