import logging
from typing import Dict, Any
from src.credit_risk_analysis.dashboard.services.model_service \
    import ModelService

# Configuración de logging para ver los errores en consola
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestService")


def test_inference_flow() -> None:
    """Prueba el flujo completo desde la entrada hasta SHAP."""
    service = ModelService()

    print("\n--- 1. Inicializando Servicio ---")
    try:
        service.initialize()
        print("✅ Servicio inicializado correctamente.")
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        return

    # Payload de prueba similar al que envía Streamlit
    mock_payload: Dict[str, Any] = {
        "age": 33,
        "sex": "male",
        "job": 2,
        "housing": "own",
        "saving_accounts": "little",
        "checking_account": "moderate",
        "credit_amount": 2500,
        "duration": 12,
        "purpose": "car",
        "inst_ratio": 0.0,
        "age_group": "adult"
    }

    print("\n--- 2. Ejecutando Predicción ---")
    # Las variables se usan AQUÍ, donde están definidas (dentro de la función)
    result = service.predict(mock_payload)

    if result.get("status") == "success":
        print("✅ Predicción exitosa!")
        print(f"   Probabilidad: {result['probability']}")
        print(f"   Decisión: {result['decision']}")
        print(f"   Valores SHAP (primeros 3): {result['shap_values'][:3]}")
    else:
        print(f"❌ Fallo en la predicción: {result.get('message')}")


if __name__ == "__main__":
    test_inference_flow()
