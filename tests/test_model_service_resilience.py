from unittest.mock import patch, MagicMock
from typing import Dict
from dashboard.services.model_service import ModelService


@patch("mlflow.get_experiment_by_name")
def test_get_metrics_server_down(mock_get_exp: MagicMock) -> None:
    """
    Verifica que el servicio sea resiliente si MLflow no responde.
    No debe lanzar excepciones, solo retornar un dict vacío [cite: 2026-02-27].
    """
    # Simulamos que la llamada a MLflow falla por red
    mock_get_exp.side_effect = Exception("MLflow Server Unreachable")

    service: ModelService = ModelService()
    # Forzamos la ejecución de get_metrics
    metrics: Dict[str, float] = service.get_metrics()

    # Validaciones
    assert isinstance(metrics, dict)
    assert len(metrics) == 0
    print("✅ Resiliencia validada: El servicio manejó la caída de MLflow.")


@patch("mlflow.search_runs")
@patch("mlflow.get_experiment_by_name")
def test_get_metrics_no_runs_found(
    mock_get_exp: MagicMock,
    mock_search: MagicMock
) -> None:
    """Verifica el comportamiento, el experimento existe pero no hay runs."""
    # Simulamos experimento encontrado pero sin ejecuciones
    mock_exp = MagicMock()
    mock_exp.experiment_id = "123"
    mock_get_exp.return_value = mock_exp

    # Simulamos DataFrame vacío (comportamiento de mlflow.search_runs)
    import pandas as pd
    mock_search.return_value = pd.DataFrame()

    service: ModelService = ModelService()
    metrics: Dict[str, float] = service.get_metrics()

    assert metrics == {}
