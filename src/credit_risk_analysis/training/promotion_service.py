from __future__ import annotations
from typing import List, Any, Dict
from mlflow.tracking import MlflowClient

# 1. Definición de alias de tipo [cite: 2026-03-04]
MetricsDict = Dict[str, Any]


class ExperimentRanker:
    """Clase para el ranking de experimentos [cite: 2026-02-27]."""

    def __init__(self, experiments_path: str) -> None:
        self.path = experiments_path

    def rank(self, metric: str) -> List[MetricsDict]:
        """Calcula ranking basado en la métrica [cite: 2026-03-04]."""
        print(f"DEBUG: Ranking con métrica: {metric}")
        return []


def promote_best_model_to_production(
    model_name: str,
    metric: str = "roc_auc",
    experiments_dir: str = "experiments"
) -> None:
    """
    Promociona el mejor experimento a MLflow [cite: 2026-02-27].
    """
    client: MlflowClient = MlflowClient()
    ranker: ExperimentRanker = ExperimentRanker(
        experiments_path=experiments_dir
    )

    # 2. Obtener el ranking (Validado por Mypy) [cite: 2026-03-04]
    ranked_exps: List[MetricsDict] = ranker.rank(metric=metric)

    if not ranked_exps:
        print("⚠️ No se encontraron experimentos.")
        return

    # 3. Identificar el mejor run [cite: 2026-03-04]
    best_exp: MetricsDict = ranked_exps[0]
    best_run_id: str = str(best_exp.get("run_id", ""))

    if not best_run_id:
        print("❌ El ganador no tiene run_id.")
        return

    # 4. Buscar versión (Gobernanza automatizada) [cite: 2026-02-27]
    filter_str: str = f"run_id='{best_run_id}'"
    versions: List[Any] = client.search_model_versions(filter_str)

    if not versions:
        print(f"⚠️ No hay versiones para el run {best_run_id}")
        return

    latest_version: str = versions[0].version

    # 5. Transición a Production [cite: 2026-02-27]
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🚀 '{model_name}' v{latest_version} movido a PRODUCTION.")
    val_met: float = float(best_exp.get(metric, 0.0))
    print(f"📊 Métrica ganadora ({metric}): {val_met:.4f}")


if __name__ == "__main__":
    promote_best_model_to_production(model_name="CreditRiskRF_Official")
