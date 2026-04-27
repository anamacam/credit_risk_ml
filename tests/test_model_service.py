import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict
from dashboard.services.model_service import ModelService
from dashboard.exceptions import ModelNotFoundError


@pytest.fixture
def mock_mlflow_model() -> MagicMock:
    """Fixture que simula un modelo cargado de MLflow [cite: 2026-03-04]."""
    mock = MagicMock()
    # Simulamos que el modelo devuelve una lista con la probabilidad
    mock.predict.return_value = [0.25]
    return mock


@patch("mlflow.pyfunc.load_model")
def test_model_service_prediction(
    mock_load: MagicMock,
    mock_mlflow_model: MagicMock
) -> None:
    """Valida que la predicción retorne el tipo float correcto."""
    mock_load.return_value = mock_mlflow_model
    service: ModelService = ModelService()
    dummy_data: Dict[str, Any] = {"age": 30, "income": 50000}
    result: float = service.predict(dummy_data)
    # Corregido S1244: Uso de pytest.approx para comparar flotantes
    assert isinstance(result, float)
    assert result == pytest.approx(0.25)
    mock_mlflow_model.predict.assert_called_once()


@patch("mlflow.pyfunc.load_model")
def test_model_service_load_error(mock_load: MagicMock) -> None:
    """Valida que se lance la excepción personalizada si falla MLflow."""
    mock_load.side_effect = Exception("Connection failed")
    with pytest.raises(ModelNotFoundError):
        ModelService()
