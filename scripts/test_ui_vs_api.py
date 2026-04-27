"""
Compara predicciones de UI vs API para diagnosticar discrepancia.
"""

import json
from pathlib import Path

import requests


def test_same_input() -> None:
    """Prueba el mismo input en ambos endpoints."""
    # Datos de prueba
    test_data = {
        "age": 35,
        "sex": "male",
        "job": 2,
        "housing": "own",
        "saving_accounts": "little",
        "checking_account": "moderate",
        "credit_amount": 5000,
        "duration": 24,
        "purpose": "car",
        "threshold": 0.387
    }

    print("=" * 80)
    print("🔍 DIAGNÓSTICO: UI vs API")
    print("=" * 80)
    print("\n📋 Input de prueba:")
    print(json.dumps(test_data, indent=2))

    # Probar API
    api_url = "http://localhost:8000/predict"
    print(f"\n📡 PROBANDO API ({api_url})...")
    try:
        api_response = requests.post(
            api_url,
            json=test_data,
            timeout=10
        )

        if api_response.status_code == 200:
            api_result = api_response.json()
            api_prob = api_result.get("probability", 0.0)
            api_decision = api_result.get("decision", "N/A")

            print("   ✅ API Response:")
            print(f"      Probabilidad: {api_prob:.4f} "
                  f"({api_prob * 100:.2f}%)")
            print(f"      Decisión: {api_decision}")
        else:
            print(f"   ❌ Error API: {api_response.status_code}")
            print(f"   {api_response.text}")
            return

    except Exception as e:
        print(f"   ❌ Error conectando a API: {e}")
        return

    # Probar UI
    print("\n🖥️ PROBANDO UI/DASHBOARD...")
    print("   ⚠️ La UI probablemente NO tiene endpoint REST")
    print("   ⚠️ Usa Streamlit y carga modelo localmente")

    # Verificar archivos de modelo
    print("\n📦 VERIFICANDO MODELOS EN DISCO:")

    paths_to_check = [
        "artifacts/active_model/model.pkl",
        "artifacts/model/model.pkl",
        "artifacts/model.pkl",
    ]

    for path_str in paths_to_check:
        path = Path(path_str)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {path} ({size_mb:.2f} MB)")
        else:
            print(f"   ❌ {path} - NO EXISTE")

    # Instrucciones finales (PEP 8 compliant)
    print("\n" + "=" * 80)
    print("💡 DIAGNÓSTICO:")
    print("=" * 80)
    print("\nLa UI (Streamlit dashboard) probablemente:")
    print("  1. Carga un modelo LOCAL diferente al de la API")
    print("  2. NO llama a la API, hace predicción directa")
    print("\n🔧 SOLUCIÓN:")
    print("  1. Modificar la UI para que llame a la API")
    print("  2. O copiar el modelo nuevo donde la UI lo busca")
    print("\n📍 BUSCAR EN EL CÓDIGO:")
    print("  - Archivo: dashboard/app.py o similar")
    print("  - Buscar: joblib.load() o pickle.load()")
    print("  - Reemplazar con: requests.post() a la API")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    test_same_input()
