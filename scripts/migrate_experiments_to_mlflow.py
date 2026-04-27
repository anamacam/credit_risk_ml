"""
Migra experimentos de Credit Risk a MLflow con métricas completas.
Maneja desempates por AUC -> F1 -> Recencia.
"""

import json
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.sklearn
import joblib
import numpy as np


def _validate_assets(exp_path: Path):
    """Valida que el experimento tenga los archivos mínimos."""
    assets = {
        "config": exp_path / "config.json",
        "metrics": exp_path / "metrics.json",
        "model": exp_path / "model.pkl"
    }
    for name, path in assets.items():
        if not path.exists():
            print(f"   ⚠️ Sin {name} - SKIP")
            return None
    return assets


def migrate_experiment(exp_path: Path):
    """Migra un experimento individual con todas sus métricas."""
    print(f"\n📦 Procesando: {exp_path.name}")

    paths = _validate_assets(exp_path)
    if not paths:
        return None

    with open(paths["config"], 'r') as f:
        config = json.load(f)
    with open(paths["metrics"], 'r') as f:
        metrics = json.load(f)

    try:
        model = joblib.load(paths["model"])
    except Exception as e:
        print(f"   ❌ Error cargando modelo: {e}")
        return None

    model_name = config.get("model_name", "Unknown")
    timestamp = exp_path.name.split('_')[0]
    run_name = f"{model_name}_{timestamp}_migrated"

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Log de Parámetros
        for k, v in config.items():
            if k not in ['model_name', 'dataset'] and v is not None:
                mlflow.log_param(k, v)

        # 2. Log de Todas las Métricas
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.float64, np.int64)):
                mlflow.log_metric(k, float(v))

        # 3. Log de Modelo (Línea dividida para evitar E501)
        mlflow.sklearn.log_model(
            model, "model", registered_model_name="credit_risk_classifier"
        )

        # 4. Tags
        mlflow.set_tag("source", "migration")
        mlflow.set_tag("migrated_at", datetime.now().isoformat())

        print(f"   ✅ Migrado - Run ID: {run.info.run_id}")
        return {
            "exp_name": exp_path.name,
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model_name": model_name
        }


def main():
    print("=" * 80)
    print("🔄 MIGRACIÓN PROFESIONAL A MLFLOW")
    print("=" * 80)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("credit_risk_analysis")

    exp_dir = Path("experiments")
    if not exp_dir.exists():
        print(f"❌ No se encontró: {exp_dir}")
        return

    exp_dirs = [
        d for d in exp_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]
    migrated = [m for d in sorted(exp_dirs) if (m := migrate_experiment(d))]

    if not migrated:
        print("\n⚠️ No se migraron experimentos.")
        return

    # CRITERIO DE DESEMPATE: 1. AUC, 2. F1-Score, 3. Nombre
    sorted_m = sorted(
        migrated,
        key=lambda x: (
            float(x['metrics'].get('roc_auc', 0)),
            float(x['metrics'].get('f1_score', 0)),
            x['exp_name']
        ),
        reverse=True
    )

    print("\n" + "=" * 80)
    print(f"📊 TOP 5 MODELOS (Total exitosos: {len(migrated)})")
    print("-" * 80)
    print(f"{'Experimento':<40} {'AUC':<10} {'F1':<10} {'KS':<10}")
    print("-" * 80)

    for exp in sorted_m[:5]:
        m = exp['metrics']
        row = (
            f"{exp['exp_name']:<40} "
            f"{m.get('roc_auc', 0):.4f}     "
            f"{m.get('f1_score', 0):.4f}     "
            f"{m.get('ks_score', 0):.4f}"
        )
        print(row)

    best = sorted_m[0]
    artifacts_path = Path("artifacts")
    artifacts_path.mkdir(exist_ok=True)

    with open(artifacts_path / "ACTIVE_MODEL.txt", "w") as f:
        f.write(f"run_id: {best['run_id']}\nexp_name: {best['exp_name']}")

    # FIX F541/S3457: Eliminadas 'f' innecesarias en mensajes estáticos
    print("\n" + "=" * 80)
    print(f"🥇 MEJOR MODELO ACTUALIZADO: {best['exp_name']}")
    print("💾 Guardado en: artifacts/ACTIVE_MODEL.txt")
    print("🔗 MLflow UI: http://localhost:5000\n")


if __name__ == "__main__":
    main()
