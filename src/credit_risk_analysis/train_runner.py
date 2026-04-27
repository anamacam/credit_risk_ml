from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from credit_risk_analysis.training.training import build_pipeline, train


def main() -> None:
    """
    Orquestador sincronizado con training.py.
    Cumple con PEP 8 (E302, W291, W293) y SonarLint S6973.
    """
    # 1. Definición de rutas profesionales
    data_path: Path = Path("data/processed/train.csv")
    proc_dir: Path = Path("data/processed")
    cache_dir: Path = Path("artifacts/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2. Carga de datos
    df: pd.DataFrame = pd.read_csv(str(data_path))
    target_col: str = "Risk"

    X: pd.DataFrame = df.drop(columns=[target_col])
    # Mapeo consistente con la lógica de negocio
    y: pd.Series[int] = df[target_col].map({"good": 0, "bad": 1})

    # 3. Definición del modelo (Resolución SonarLint S6973)
    model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )

    # 4. Construcción del Pipeline
    pipeline: Pipeline = build_pipeline(
        model=model,
        df_x=X,
        cache_dir=cache_dir
    )

    # 5. Entrenamiento y Registro en MLflow
    _, run_id = train(
        df_x=X,
        y=y,
        pipeline=pipeline,
        proc_dir=proc_dir
    )

    print(f"🚀 Build Exitoso | MLflow Run ID: {run_id}")


if __name__ == "__main__":
    main()
