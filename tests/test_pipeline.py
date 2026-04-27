import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.trainer import validate_pipeline_features


def test_metadata_mismatch_raises_error() -> None:
    """
    Test that validation raises ValueError when features don't match.
    """
    # Arrange
    # S6969: Especificamos 'memory=None' para que SonarLint no proteste
    pipeline = Pipeline(steps=[("scaler", StandardScaler())], memory=None)

    # Creamos un DataFrame para que Scikit-Learn registre 'feature_names_in_'
    df = pd.DataFrame([[1, 2]], columns=["feat1", "feat2"])
    pipeline.fit(df, [1])

    # Lista que NO coincide con las columnas del DataFrame
    wrong_features = ["feat1", "wrong_name"]

    # Act & Assert
    # CAMBIO CRUCIAL: De 'bytes.raises' a 'pytest.raises'
    with pytest.raises(ValueError, match="Metadata features do not match"):
        validate_pipeline_features(pipeline, wrong_features)
