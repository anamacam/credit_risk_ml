from typing import List, Any, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    memory: Optional[str] = None
) -> ColumnTransformer:
    """
    Construye el pipeline de preprocesamiento con soporte para caché.
    """
    # Definición de pasos para variables numéricas
    numeric_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]
    # S6969: Se especifica el argumento memory para optimización
    numeric_pipeline: Pipeline = Pipeline(steps=numeric_steps, memory=memory)

    # Definición de pasos para variables categóricas
    categorical_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ]
    # S6969: Se especifica el argumento memory para optimización
    categorical_pipeline: Pipeline = Pipeline(
        steps=categorical_steps,
        memory=memory
    )

    # ColumnTransformer configurado para cumplir con Flake8 (E501)
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop"
    )
