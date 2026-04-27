from pathlib import Path
import pickle


# Ruta al modelo según tu model_loader
MODEL_PATH = Path("models/credit_model_v1.pkl")


def inspect_model():
    """Inspecciona las variables de entrenamiento del modelo."""
    if not MODEL_PATH.exists():
        print(f"No encontré el modelo en: {MODEL_PATH.resolve()}")
        return

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    print("-" * 30)
    print("INSPECCIÓN DEL MODELO")
    print("-" * 30)

    # 1. Intentar sacar nombres si es un modelo de Scikit-Learn
    if hasattr(model, 'feature_names_in_'):
        print("\n✅ VARIABLES DE ENTRADA DETECTADAS:")
        for i, name in enumerate(model.feature_names_in_, 1):
            print(f"{i}. {name}")

    # 2. Si es un Pipeline, inspeccionar el primer paso
    elif hasattr(model, 'steps'):
        print("\n📦 ES UN PIPELINE. Intentando ver nombres...")
        try:
            # Intentamos sacar los nombres del primer paso
            names = model.steps[0][1].get_feature_names_out()
            for i, name in enumerate(names, 1):
                print(f"{i}. {name}")
        except Exception:
            print("No se pudieron extraer nombres automáticamente.")

    # 3. Ver qué tipo de objeto es
    print(f"\nTipo de modelo: {type(model)}")
    print("-" * 30)


if __name__ == "__main__":
    inspect_model()
