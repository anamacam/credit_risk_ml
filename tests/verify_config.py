import json
from src.credit_risk_analysis.config.config_loader import config
from tests.test_load import generate_random_data


def test_integration() -> None:
    """Verifica la carga de YAML y la generación de datos."""
    print("🔍 REVISANDO CONFIGURACIÓN CARGADA...")
    metadata = config.get("project_metadata", {})

    print(f"✅ Proyecto detectado: {metadata.get('name', 'N/A')}")

    print("\n🎲 GENERANDO PAYLOAD DE PRUEBA...")
    sample_data = generate_random_data()

    if sample_data:
        print("✅ SUCCESS: Payload generado correctamente.")
        print(json.dumps(sample_data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    test_integration()
