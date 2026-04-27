from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, cast

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Configuración de logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mlflow_config() -> None:
    """Configura MLflow usando SQLite y define el experimento."""
    base_dir: Path = Path(__file__).resolve().parents[3]
    db_path: Path = base_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    if mlflow.active_run() is None:
        mlflow.set_experiment("credit_risk_analysis_training")


def prepare_data(path: Path) -> Tuple[pd.DataFrame, pd.Series[Any]]:
    """Carga datos y aplica OHE. CORRECCIÓN: pd.Series[Any] para Mypy."""
    df: pd.DataFrame = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["risk"] = df["Risk"].map({"good": 0, "bad": 1})
    df = df.drop(columns=["Risk"])
    df = pd.get_dummies(df, drop_first=True)

    x_data: pd.DataFrame = df.drop(columns=["risk"])
    y_data: pd.Series[Any] = df["risk"]
    return x_data, y_data


def save_governance_docs(metrics: Dict[str, float]) -> None:
    """Genera README y métricas para auditoría de gobernanza."""
    readme_content: str = (
        "# Model Card: Credit Risk RF\n"
        "## Objetivo\nPredecir impagos bancarios.\n"
        "## XAI\nExplicaciones SHAP enriquecidas.\n"
        "## Governance\nModelo auditable con firma y métricas."
    )
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def train_model(
    data_path: Path,
    n_estimators: int = 200,
    max_depth: int = 12,
    min_samples_leaf: int = 5
) -> Dict[str, Any]:
    """Pipeline con métricas completas y XAI robusto."""
    get_mlflow_config()
    x_data, y_data = prepare_data(data_path)

    # CORRECCIÓN: Tipado de Series en el split
    res_split = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    x_train, x_test, y_train, y_test = res_split

    with mlflow.start_run(run_name="Credit_Risk_XAI_Final") as r:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced"
        )
        model.fit(x_train, y_train)

        # 1. EVALUACIÓN (Tipado corregido para Mypy)
        y_prob: npt.NDArray[np.float64] = model.predict_proba(x_test)[:, 1]
        y_pred: npt.NDArray[np.int64] = (y_prob > 0.5).astype(np.int64)

        metrics: Dict[str, float] = {
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred))
        }
        mlflow.log_metrics(metrics)

        # 2. XAI (SHAP) - Resolución de dimensiones
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(x_test)

        if isinstance(shap_vals, list):
            p_shap: npt.NDArray[np.float64] = np.array(shap_vals[1])
        elif len(shap_vals.shape) == 3:
            p_shap = shap_vals[:, :, 1]
        else:
            p_shap = cast(npt.NDArray[np.float64], shap_vals)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(p_shap, x_test, show=False)
        plt.savefig("shap_summary.png", bbox_inches="tight")
        mlflow.log_artifact("shap_summary.png", "XAI_Global")
        plt.close()

        # 3. REPORTE DETALLADO (Lógica de importancia)
        f_names: List[str] = x_test.columns.tolist()
        imp_vec: npt.NDArray[np.float64] = (
            np.abs(p_shap).mean(axis=0).flatten()
        )
        detailed_report: Dict[str, Any] = {}

        for i, name in enumerate(f_names):
            feat_v: npt.NDArray[Any] = np.array(x_test.iloc[:, i]).flatten()
            s_v: npt.NDArray[np.float64] = np.array(p_shap[:, i]).flatten()
            c_val: float = 0.0
            if feat_v.shape[0] == s_v.shape[0]:
                c_matrix = np.corrcoef(feat_v.astype(float), s_v)
                c_val = float(c_matrix[0, 1]) if c_matrix.size > 1 else 0.0

            detailed_report[name] = {
                "magnitude": float(imp_vec[i]),
                "direction": "Positive" if c_val > 0 else "Negative",
                "correlation": round(c_val, 4)
            }

        sorted_rep = dict(
            sorted(
                detailed_report.items(),
                key=lambda x: x[1]["magnitude"],
                reverse=True
            )
        )
        with open("shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(sorted_rep, f, indent=4)
        mlflow.log_artifact("shap_summary.json", "XAI_Global")

        # 4. EXPORTACIÓN Y REGISTRY
        artifact_dir = Path("artifacts/model")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact_dir / "model.pkl", compress=3)

        signature = infer_signature(x_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CreditRiskRF_Official",
            signature=signature,
            input_example=x_test.head(3)
        )

        save_governance_docs(metrics)
        return {"run_id": str(r.info.run_id)}


def main() -> None:
    """Main con resolución de rutas dinámica."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=5)

    args: argparse.Namespace = parser.parse_args()
    r_path: Path = Path(args.data_path)

    if not r_path.is_absolute():
        f_path: Path = Path(__file__).resolve().parents[3] / r_path
    else:
        f_path = r_path

    if not f_path.exists():
        raise FileNotFoundError(f"❌ Error: {f_path} no encontrado.")

    res = train_model(
        data_path=f_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf
    )
    print(f"✅ Éxito. Run ID: {res['run_id']}")


if __name__ == "__main__":
    main()
