from typing import Dict, Any, Tuple, Optional, Type
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

# Tipado estricto para evitar errores de definición de Mypy
MODEL_REGISTRY: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "random_forest": (
        "RandomForestClassifier",
        {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": 42,
            "class_weight": "balanced",
        },
    ),
    "decision_tree": (
        "DecisionTreeClassifier",
        {
            "max_depth": 10,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "random_state": 42,
            "class_weight": "balanced",
        },
    ),
    "logistic_regression": (
        "LogisticRegression",
        {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
        },
    ),
}


def build_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None
) -> BaseEstimator:
    """
    Factory de modelos profesional con tipado completo.

    Args:
        model_name: Identificador del modelo en el registry.
        params: Diccionario opcional para sobrescribir hiperparámetros.

    Returns:
        Estimador de Scikit-learn configurado.
    """
    if model_name not in MODEL_REGISTRY:
        available: str = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Modelo no soportado: {model_name}. Use: {available}"
        )

    model_type: str
    default_params: Dict[str, Any]
    model_type, default_params = MODEL_REGISTRY[model_name]

    # Combinar defaults + params (params tiene prioridad)
    final_params: Dict[str, Any] = default_params.copy()
    if params is not None:
        final_params.update(params)

    # Mapa de clases explícito con tipado Type para mayor seguridad
    model_classes: Dict[str, Type[BaseEstimator]] = {
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "LogisticRegression": LogisticRegression,
    }

    model_class: Type[BaseEstimator] = model_classes[model_type]

    # E501: Desempaquetado limpio de argumentos
    model: BaseEstimator = model_class(**final_params)
    return model
