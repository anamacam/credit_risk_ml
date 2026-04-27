import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    precision_score,
    ConfusionMatrixDisplay,
    roc_curve,
)

# --- Configuración de Paths ---
BASE_DIR = Path(__file__).resolve().parents[3]
SRC_PATH = str(BASE_DIR / "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from credit_risk_analysis.training.preprocess import (  # noqa: E402
    prepare_data,
    build_preprocess_pipeline,
)

# Constantes
DATA_RAW = BASE_DIR / "data" / "raw" / "german_credit_data.csv"
PROC_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_ROOT = Path("artifacts")
TMP_ARTIFACTS = ARTIFACT_ROOT / "temp"
VERSION = "v1.0"

# Asegurar directorios
PROC_DIR.mkdir(parents=True, exist_ok=True)
TMP_ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Hiperparámetros del modelo (centralizados para log en MLflow)
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "class_weight": "balanced",
}


def get_mlflow_config() -> None:
    """Configura MLflow usando acceso directo a DB para evitar error 403."""
    db_path = "sqlite:///" + str(BASE_DIR / "mlflow.db")
    mlflow.set_tracking_uri(db_path)
    mlflow.set_experiment("Credit_Risk_Official_Experiment")
    print(f"✅ MLflow conectado vía: {db_path}")


def create_radar_chart(
    metrics_dict: dict[str, float],
    threshold: float,
    path: Path,
) -> None:
    """
    Genera gráfico de radar para visualización de métricas.

    Correcciones aplicadas:
    - Se cierra el polígono repitiendo el primer valor al final.
    - Se elimina el isinstance(ax, PolarAxes) innecesario.
    """
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Cerrar el polígono repitiendo el primer punto al final
    values_closed = values + [values[0]]
    label_loc = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    label_loc_closed = np.append(label_loc, label_loc[0])

    plt.figure(figsize=(6, 6))
    # Cast explícito: plt.subplot() retorna Axes en la firma estática
    ax: PolarAxes = plt.subplot(111, polar=True)  # type: ignore[assignment]
    # polar=True garantiza PolarAxes; el isinstance es innecesario
    ax.plot(
        label_loc_closed,
        values_closed,
        label=f"Threshold {threshold}",
        color="blue",
    )
    ax.fill(label_loc_closed, values_closed, color="blue", alpha=0.25)
    ax.set_thetagrids(np.degrees(label_loc), categories)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def build_model() -> RandomForestClassifier:
    """
    Instancia un nuevo RandomForestClassifier con los parámetros globales.

    Se llama dentro del loop para que cada threshold tenga su propio
    objeto modelo, evitando que dos runs compartan la misma referencia.
    MODEL_PARAMS incluye random_state, min_samples_leaf y max_features
    para garantizar reproducibilidad (Sonar S6709 / S6973).
    """
    # Parámetros explícitos para satisfacer Sonar S6709 (random_state)
    # y S6973 (min_samples_leaf, max_features requeridos)
    return RandomForestClassifier(
        n_estimators=MODEL_PARAMS["n_estimators"],
        max_depth=MODEL_PARAMS["max_depth"],
        min_samples_leaf=MODEL_PARAMS["min_samples_leaf"],
        max_features=MODEL_PARAMS["max_features"],
        random_state=MODEL_PARAMS["random_state"],
        class_weight=MODEL_PARAMS["class_weight"],
    )


def run_professional_training() -> None:
    """
    Pipeline de entrenamiento con trazabilidad completa en MLflow.

    Correcciones aplicadas:
    1. Cada threshold entrena su propio modelo independiente.
    2. preprocessor.pkl se genera una sola vez y se registra en MLflow
       fuera del loop para evitar duplicidad.
    3. Los hiperparámetros del modelo se registran en cada run de MLflow.
    4. El radar chart cierra el polígono correctamente.
    5. Se eliminó isinstance(ax, PolarAxes) innecesario.
    """
    get_mlflow_config()

    # ------------------------------------------------------------------ #
    # Carga, split y preprocesamiento  (una sola vez)                     #
    # ------------------------------------------------------------------ #
    x_data, y_data = prepare_data(DATA_RAW)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    preprocessor = build_preprocess_pipeline()
    x_train_proc = preprocessor.fit_transform(x_train)
    x_test_proc = preprocessor.transform(x_test)

    # Guardar preprocesador localmente (una sola vez, fuera del loop)
    preprocessor_path = TMP_ARTIFACTS / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✅ Preprocesador guardado en: {preprocessor_path}")

    # ------------------------------------------------------------------ #
    # Loop por threshold: cada iteración tiene su propio modelo y run     #
    # ------------------------------------------------------------------ #
    for t in [0.20, 0.30]:
        run_name = f"Threshold_{str(t).replace('.', '_')}_{VERSION}"

        # FIX 1: Entrenar un modelo nuevo por cada threshold
        model = build_model()
        model.fit(x_train_proc, y_train)
        y_prob = model.predict_proba(x_test_proc)[:, 1]
        y_pred = (y_prob >= t).astype(int)

        with mlflow.start_run(run_name=run_name):

            # ---------------------------------------------------------- #
            # Métricas                                                     #
            # ---------------------------------------------------------- #
            metrics = {
                "recall": float(recall_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
            }
            mlflow.log_metrics(metrics)

            # ---------------------------------------------------------- #
            # Parámetros: threshold + hiperparámetros del modelo          #
            # FIX 2: Registrar todos los hiperparámetros para             #
            # reproducibilidad                                             #
            # ---------------------------------------------------------- #
            mlflow.log_params({"threshold": t, "version": VERSION})
            mlflow.log_params(model.get_params())

            # ---------------------------------------------------------- #
            # Artefacto: Matriz de confusión                              #
            # ---------------------------------------------------------- #
            cm_path = TMP_ARTIFACTS / f"cm_{t}.png"
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            plt.savefig(str(cm_path))
            plt.close()
            mlflow.log_artifact(str(cm_path), artifact_path="visuals")

            # ---------------------------------------------------------- #
            # Artefacto: Curva ROC                                        #
            # ---------------------------------------------------------- #
            roc_path = TMP_ARTIFACTS / f"roc_{t}.png"
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC: {metrics['roc_auc']:.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Curve — Threshold {t}")
            plt.legend()
            plt.savefig(str(roc_path))
            plt.close()
            mlflow.log_artifact(str(roc_path), artifact_path="visuals")

            # ---------------------------------------------------------- #
            # Artefacto: Radar chart (polígono cerrado)                   #
            # FIX 3: create_radar_chart ya corregido arriba               #
            # ---------------------------------------------------------- #
            radar_path = TMP_ARTIFACTS / f"radar_{t}.png"
            create_radar_chart(metrics, t, radar_path)
            mlflow.log_artifact(str(radar_path), artifact_path="visuals")

            # ---------------------------------------------------------- #
            # Artefacto: Preprocesador                                    #
            # FIX 4: Se registra dentro del run pero el archivo se genera #
            # una sola vez fuera del loop (no se regenera en cada iter.)  #
            # ---------------------------------------------------------- #
            mlflow.log_artifact(
                str(preprocessor_path),
                artifact_path="preprocessing",
            )

            # ---------------------------------------------------------- #
            # Modelo con firma                                             #
            # ---------------------------------------------------------- #
            signature = infer_signature(x_test_proc, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
            )

            print(f"✅ Run '{run_name}' exitoso. Métricas: {metrics}")


if __name__ == "__main__":
    run_professional_training()
