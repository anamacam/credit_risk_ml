import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.trainer import validate_pipeline_features


def test_metadata_mismatch_raises_error() -> None:
    """
    Verifica que la validación lanza ValueError si las features no coinciden.
    """
    # Arrange
    # S6969: Especificamos 'memory=None' para satisfacer a SonarLint
    pipeline = Pipeline(steps=[("scaler", StandardScaler())], memory=None)

    # Entrena con un DataFrame para que Scikit-Learn genere 'feature_names_in'
    df = pd.DataFrame([[1, 2]], columns=["feat1", "feat2"])
    pipeline.fit(df, [1])

    # Definimos columnas que NO coinciden
    wrong_features = ["feat1", "wrong_name"]

    # Act & Assert
    # CAMBIO CRUCIAL: Se reemplaza 'bytes.raises' por 'pytest.raises'
    with pytest.raises(ValueError, match="Metadata features do not match"):
        validate_pipeline_features(pipeline, wrong_features)
