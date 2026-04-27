import os
from typing import Any

import joblib
from dotenv import load_dotenv

load_dotenv()


def _print_categories(transformer_obj: Any) -> None:
    """Imprime las categorias de un ColumnTransformer."""
    for name, trans, columns in transformer_obj.transformers_:
        if name != 'remainder' and hasattr(trans, 'categories_'):
            for i, col in enumerate(columns):
                cats = trans.categories_[i].tolist()
                print(f"Columna: {col} | Categorias: {cats}")


def inspect() -> None:
    """Inspecciona el contenido de los artefactos (.pkl)."""
    model_path = os.getenv("MODEL_PATH", "artifacts/model/model.pkl")
    # Buscamos el preprocesador en la ruta estandar
    prep_path = "artifacts/preprocessor.pkl"

    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encuentra el modelo en {model_path}")
        return

    obj = joblib.load(model_path)
    print(f"--- Analizando: {model_path} ---")
    print(f"Tipo detectado: {type(obj)}")

    # Si es solo el RandomForest, buscamos el preprocesador aparte
    if not isinstance(obj, dict) and not hasattr(obj, "named_steps"):
        print("⚠️ El modelo es un estimador puro. Buscando preprocesador...")
        if os.path.exists(prep_path):
            prep_obj = joblib.load(prep_path)
            _print_categories(prep_obj)
        else:
            print(f"❌ No se encontro {prep_path}. Revisa tus artefactos.")
    elif isinstance(obj, dict):
        prep = obj.get("preprocessor")
        if prep:
            _print_categories(prep)


if __name__ == "__main__":
    inspect()
