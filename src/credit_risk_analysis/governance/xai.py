# src/governance/xai.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import shap
import pandas as pd
from sklearn.pipeline import Pipeline


def log_xai_explanations(pipeline: Pipeline, x_test: pd.DataFrame) -> None:
    """
    Genera gráfico SHAP summary y lo registra como artefacto en MLflow.
    Capa de XAI manual (no forma parte del pipeline sklearn).
    """
    print("🧬 Generating SHAP explanations...")

    model = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["prep"]

    x_test_transformed = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test_transformed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        x_test_transformed,
        feature_names=feature_names,
        show=False,
    )

    plot_path = "shap_summary.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(plot_path, "plots/xai")
    plt.close()

    path_obj = Path(plot_path)
    if path_obj.exists():
        path_obj.unlink()
