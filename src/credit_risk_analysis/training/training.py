from typing import Tuple, List, Any
from pathlib import Path
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import Memory
from sklearn.metrics import roc_auc_score


def get_mlflow_config() -> None:
    """Configuración con ruta absoluta para evitar experimentos vacíos."""
    project_root: Path = Path(__file__).resolve().parents[2]
    db_path: Path = project_root / "mlflow.db"

    tracking_uri: str = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit_risk_analysis")

    mlflow.sklearn.autolog(log_models=False, silent=True)


def build_pipeline(
    model: ClassifierMixin,
    df_x: pd.DataFrame,
    cache_dir: Path,
) -> Pipeline:
    """Pipeline profesional que separa numéricas y categóricas."""
    memory: Memory = Memory(location=str(cache_dir), verbose=0)

    num_cols: List[str] = df_x.select_dtypes(
        include=['number']
    ).columns.tolist()
    cat_cols: List[str] = df_x.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ],
        memory=memory
    )


def train(
    df_x: pd.DataFrame,
    y: pd.Series[Any],
    pipeline: Pipeline,
    proc_dir: Path
) -> Tuple[Pipeline, str]:
    """Entrenamiento con registro de artefactos y modelo."""
    get_mlflow_config()

    x_train: pd.DataFrame = df_x.copy()
    for col in x_train.select_dtypes(include=['int64']).columns:
        x_train[col] = x_train[col].astype('float64')

    with mlflow.start_run(run_name="credit_risk_train") as run:
        pipeline.fit(x_train, y)

        y_proba: Any = pipeline.predict_proba(x_train)[:, 1]
        roc_auc: float = float(roc_auc_score(y, y_proba))
        mlflow.log_metric("roc_auc_train", roc_auc)

        # 1. Registro de TODA la carpeta processed para gobernanza total
        if proc_dir.exists():
            # Subimos todo (X_train, y_train, pkls, csvs) a una carpeta central
            mlflow.log_artifacts(str(proc_dir), artifact_path="processed_data")

            # 2. Registro redundante de preprocessor para fácil acceso
            pre_file: Path = proc_dir / "preprocessor.pkl"
            if pre_file.exists():
                mlflow.log_artifact(str(pre_file), "preprocessing")

            # 3. Registro redundante del clean_data para data_snapshot
            clean_csv: Path = proc_dir / "clean_data.csv"
            if clean_csv.exists():
                mlflow.log_artifact(str(clean_csv), "data_snapshot")

        # 4. Registro formal del Modelo para el Model Registry
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=x_train.head(3),
            registered_model_name="CreditRiskRF_Official"
        )

        run_id: str = str(run.info.run_id)
        print(f"✅ Run ID: {run_id} | ROC AUC: {roc_auc:.4f}")

    return pipeline, run_id


def main(
    data_path: str,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int
) -> None:
    """Entry point con tipado estricto y resolución de rutas."""
    # Resolvemos la ruta absoluta para que no falle en Windows
    input_path: Path = Path(data_path).resolve()
    # parents[1] sube de raw/ a data/ para encontrar processed/
    proc_dir: Path = input_path.parents[1] / "processed"
    cache_dir: Path = Path("artifacts/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = pd.read_csv(str(input_path))
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    df_x: pd.DataFrame = df.drop(columns=["Risk"])
    y: pd.Series[Any] = df["Risk"].map({"good": 0, "bad": 1})

    model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )

    pipeline: Pipeline = build_pipeline(model, df_x, cache_dir)
    train(df_x, y, pipeline, proc_dir)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=5)

    args: argparse.Namespace = parser.parse_args()
    main(
        data_path=str(args.data_path),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf)
    )
