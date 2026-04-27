from __future__ import annotations
from mlflow.tracking import MlflowClient


def update_registry_documentation(model_name: str, version: str) -> None:
    """
    Actualiza la metadata del modelo y de una versión concreta
    en el Model Registry (capa de governance).
    """
    client: MlflowClient = MlflowClient()

    # E501: Usamos f-strings multilínea para no superar los 79 caracteres
    description_text: str = (
        "# Credit Risk Classification Model\n\n"
        "Enterprise RandomForest Pipeline for credit default prediction.\n\n"
        "Architecture:\n"
        "- OneHotEncoding for categorical variables\n"
        "- Numerical passthrough\n"
        "- Deterministic training (random_state=42)\n\n"
        "**XAI Enabled:** Native MLflow Evaluation + SHAP Artifacts."
    )

    # Actualización del repositorio general del modelo
    client.update_registered_model(
        name=model_name,
        description="Repository for Credit Risk Enterprise models.",
    )

    # Actualización de la versión específica con la documentación técnica
    client.update_model_version(
        name=model_name,
        version=version,
        description=description_text,
    )

    print(f"📘 Documentation injected into {model_name} v{version}")
