import os
from typing import cast, List
import pandas as pd
import mlflow
from mlflow.entities import Experiment

# Constantes globales
TRACKING_URI = "http://localhost:5000"


def find_my_run() -> None:
    """Busca un run en MLflow cumpliendo reglas estrictas de Mypy/Flake8."""
    # 1. Configurar conexión
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"🔗 Conectado a: {TRACKING_URI}")

    # 2. Declaración y asignación inmediata
    # Esto elimina el error 'Name not defined' en Mypy
    all_exps: List[Experiment] = cast(
        List[Experiment],
        mlflow.search_experiments()
    )
    filter_q: str = "attributes.run_id LIKE '%f6ec%'"

    print("\n📋 EXPERIMENTOS del server:")
    for exp in all_exps:
        print(f"  {exp.experiment_id}: {exp.name}")

    print("\n🔍 BUSCANDO f6ec... en TODOS:")

    # Extraemos IDs en una lista propia para limpiar el search_runs
    exp_ids: List[str] = [e.experiment_id for e in all_exps]

    # 3. Búsqueda de runs
    raw_results = mlflow.search_runs(
        experiment_ids=exp_ids,
        filter_string=filter_q
    )

    # 4. Forzar el tipo DataFrame para evitar 'union-attr'
    results = cast(pd.DataFrame, raw_results)

    if results.empty:
        print("\n❌ NO HAY run f6ec... en el server")
        return

    for _, run in results.iterrows():
        # Variables cortas para cumplir PEP 8 (max 79 chars)
        r_id = str(run.get("run_id", "N/A"))
        e_id = str(run.get("experiment_id", "N/A"))
        print(f"🎯 ENCONTRADO: Exp {e_id} -> {r_id}")


if __name__ == "__main__":
    find_my_run()
