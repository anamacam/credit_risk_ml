#!/usr/bin/env python
"""Inspecciona el contenido del preprocesador."""

import sys
from pathlib import Path

import joblib


def check_preprocessor():
    """Verifica el contenido del preprocesador."""
    preprocessor_path = Path(
        "experiments/exp_20260322_155648_v3_optimized/preprocessor.pkl"
    )

    if not preprocessor_path.exists():
        print(f"❌ No se encuentra: {preprocessor_path}")
        return False

    try:
        print(f"📦 Cargando preprocesador desde: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)

        print("\n✅ Preprocesador cargado exitosamente")
        print(f"   Tipo: {type(preprocessor).__name__}")

        # Inspeccionar según el tipo
        if hasattr(preprocessor, 'named_steps'):
            steps = list(preprocessor.named_steps.keys())
            print(f"   Es un Pipeline con steps: {steps}")

        elif hasattr(preprocessor, 'transformers_'):
            transformers = len(preprocessor.transformers_)
            print(f"   Es ColumnTransformer con {transformers} transformers")

        elif hasattr(preprocessor, 'get_feature_names_out'):
            try:
                features = preprocessor.get_feature_names_out()
                print(f"  Features: {features[:5]}... ({len(features)} total)")
            except AttributeError:
                print("   No se pudieron obtener los nombres de features")

        # Tamaño en memoria
        size_bytes = sys.getsizeof(preprocessor)
        print(f"   Tamaño en memoria: {size_bytes} bytes")

        return True

    except Exception as e:
        print(f"❌ Error al cargar preprocesador: {e}")
        return False


def check_model():
    """Verifica el contenido del modelo."""
    model_path = Path(
        "experiments/exp_20260322_155648_v3_optimized/model.pkl"
    )

    if not model_path.exists():
        print(f"❌ No se encuentra: {model_path}")
        return False

    try:
        print("\n📦 Cargando modelo desde: {model_path}")
        model = joblib.load(model_path)

        print("\n✅ Modelo cargado exitosamente")
        print(f"   Tipo: {type(model).__name__}")

        if hasattr(model, 'named_steps'):
            steps = list(model.named_steps.keys())
            print(f"   Es un Pipeline con steps: {steps}")

            # Verificar si tiene preprocesador dentro del pipeline
            if 'preprocessor' in model.named_steps:
                print("   ✅ El modelo ya incluye el preprocesador")

        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            print(f"   Features esperadas: {len(features)}")
            print(f"   Primeras 5: {features[:5]}")

        return True

    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return False


def main():
    """Función principal."""
    print("🔍 Verificando artefactos del modelo")
    print("=" * 50)

    preprocessor_ok = check_preprocessor()
    model_ok = check_model()

    print("\n" + "=" * 50)
    if preprocessor_ok and model_ok:
        print("✅ Todos los artefactos son válidos")
    else:
        print("⚠️ Hay problemas con los artefactos")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
