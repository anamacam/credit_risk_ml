"""
Automated test script for the Credit Risk API.
"""

import requests


# Configuración: Usamos 127.0.0.1 para conectar desde el cliente a la API
URL = "http://127.0.0.1:8000/predict"


def test_single_prediction() -> None:
    """
    Envía un caso de prueba a la API y verifica la respuesta.
    """
    # Estructura anidada bajo la llave "data" requerida por la API
    payload = {
        "data": {
            "Age": 33,
            "Sex": "male",
            "Housing": "own",
            "Saving accounts": "little",
            "Checking account": "quite rich",
            "Credit amount": 2500,
            "Duration": 24,
            "Purpose": "radio/TV"
        }
    }

    print(f"🚀 Enviando solicitud a: {URL}...")

    try:
        # Realizamos la petición POST enviando el JSON
        response = requests.post(
            URL,
            json=payload,
            timeout=10
        )

        # Verificamos si la API respondió con éxito (200 OK)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prueba exitosa!")
            print(f"   Predicción: {result['prediction']}")

            # Verificación de metadatos opcionales
            if "status" in result:
                print(f"   Estado: {result['status']}")
        else:
            # En caso de error, imprimimos el detalle para diagnóstico
            error_msg = f"❌ Error {response.status_code}: {response.text}"
            print(error_msg)

    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar a la API.")
        print("   Asegúrate de que 'python src/api/main.py' esté corriendo.")


if __name__ == "__main__":
    test_single_prediction()
