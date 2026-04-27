from pathlib import Path

# Sustituye 'src.models.registry' por la ruta real de tu proyecto
from src.models.registry import ModelRegistry


def test_registry_initialization():
    """
    Fix for F821: Ensures Path and ModelRegistry are defined via imports.
    """
    # Arrange
    base_path = Path("models/artifacts")

    # Act
    registry = ModelRegistry(base_path=base_path)

    # Assert
    assert registry.base_path == base_path


def test_pipeline_build_structure():
    """
    Verifica la estructura básica del pipeline.
    """
    # Nota: Aquí deberías importar y usar tu builder
    # de forma similar a los pasos anteriores.
    pass
