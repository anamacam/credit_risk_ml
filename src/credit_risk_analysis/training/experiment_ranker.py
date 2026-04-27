from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Definición de tipo para claridad y consistencia
MetricsDict = Dict[str, Any]


class ExperimentRanker:
    """Orquestador para clasificar experimentos basado en métricas JSON."""

    def __init__(self, experiments_path: str = "experiments") -> None:
        # Mypy: Anotación de atributo de instancia
        self.experiments_path: Path = Path(experiments_path)

    def _load_metrics(
        self,
        exp_path: Path,
    ) -> Optional[MetricsDict]:
        """Carga y valida el archivo de métricas de un experimento."""
        metrics_file: Path = exp_path / "metrics.json"

        if not metrics_file.exists():
            return None

        with metrics_file.open("r", encoding="utf-8") as f:
            raw_data: Any = json.load(f)

        if not isinstance(raw_data, dict):
            return None

        # Clonamos para evitar efectos secundarios y añadimos metadata
        metrics: MetricsDict = raw_data.copy()
        metrics["experiment_name"] = exp_path.name

        return metrics

    def rank(
        self,
        metric: str = "roc_auc",
        descending: bool = True,
    ) -> List[MetricsDict]:
        """Genera una lista ordenada de experimentos por una métrica dada."""
        experiments: List[MetricsDict] = []

        # Verificamos que el path exista antes de iterar
        if not self.experiments_path.exists():
            return []

        for exp_dir in self.experiments_path.iterdir():
            if not exp_dir.is_dir():
                continue

            metrics: Optional[MetricsDict] = self._load_metrics(exp_dir)

            if metrics is None or metric not in metrics:
                continue

            experiments.append(metrics)

        # E501: Formateo de lambda para no exceder 79 caracteres
        return sorted(
            experiments,
            key=lambda x: float(x[metric]),
            reverse=descending,
        )

    def print_ranking(
        self,
        metric: str = "roc_auc",
    ) -> None:
        """Imprime en consola el ranking detallado de modelos."""
        ranked: List[MetricsDict] = self.rank(metric=metric)

        if not ranked:
            print("No experiments found.")
            return

        print(f"\n📊 Ranking by {metric}\n" + "-" * 40)

        for idx, exp in enumerate(ranked, start=1):
            score: float = float(exp[metric])
            # Manejo de tipos para evitar errores de f-string con 'N/A'
            accuracy: Union[float, str] = exp.get("accuracy", "N/A")
            acc_str: str = (
                f"{accuracy:.4f}" if isinstance(accuracy, float) else "N/A"
            )

            print(
                f"{idx}. {exp['experiment_name']} | "
                f"{metric}: {score:.4f} | "
                f"accuracy: {acc_str}"
            )

        best_name: str = str(ranked[0]["experiment_name"])
        print(f"\n🏆 Best experiment: {best_name}")
