"""
Suite completa de validación del modelo de riesgo crediticio.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest
import shap
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# .parents[1] es la carpeta 'tests/'
# .parents[2] es la raíz del proyecto 'credit_risk_ml/'
ROOT_DIR = Path(__file__).resolve().parents[1]

ARTIFACTS = ROOT_DIR / "artifacts"
DATA_RAW = ROOT_DIR / "data" / "raw" / "german_credit_data.csv"


@pytest.fixture(scope="session")
def model() -> Any:
    """Carga el modelo entrenado."""
    model_path = ARTIFACTS / "model" / "model.pkl"
    assert model_path.exists(), f"Modelo no encontrado: {model_path}"
    with open(model_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def preprocessor() -> Any:
    """Carga el preprocesador."""
    path = ARTIFACTS / "preprocessor.pkl"
    assert path.exists(), f"Preprocesador no encontrado: {path}"
    return joblib.load(path)


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """Carga el dataset original."""
    assert DATA_RAW.exists(), f"Dataset no encontrado: {DATA_RAW}"
    df = pd.read_csv(DATA_RAW, index_col=0)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


@pytest.fixture(scope="session")
def x_y(
    dataset: pd.DataFrame,
    preprocessor: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforma los datos y retorna matrices (X, y) de NumPy."""
    target_col = "risk"
    # 1. np.asarray elimina la ambigüedad del 'ExtensionArray' (Mypy line 69)
    y = np.asarray((dataset[target_col] == "bad").astype(int))
    x_raw = dataset.drop(columns=[target_col])
    # Aseguramos que x_data también sea un ndarray puro
    x_data = np.asarray(preprocessor.transform(x_raw))
    return x_data, y


@pytest.fixture(scope="session")
def y_prob(model: Any, x_y: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Probabilidades predichas por el modelo."""
    x_data: np.ndarray = np.asarray(x_y[0])

    # Realiza la predicción y extraemos la columna de clase positiva (1)
    # model.predict_proba devuelve un ndarray de forma (n_samples, 2)
    return model.predict_proba(x_data)[:, 1]


@pytest.fixture(scope="session")
def baseline_scores() -> np.ndarray:
    """Scores de referencia del entrenamiento."""
    path = ARTIFACTS / "baseline_stats.pkl"
    if path.exists():
        return joblib.load(path)
    pytest.skip("baseline_stats.pkl no encontrado — omitiendo PSI")


# ---------------------------------------------------------------------------
# 1. PRUEBAS DE RENDIMIENTO
# ---------------------------------------------------------------------------
class TestRendimiento:
    """Valida umbrales mínimos de discriminación."""

    AUC_MIN: float = 0.75
    GINI_MIN: float = 0.50
    KS_MIN: float = 0.30

    def test_roc_auc(
        self, y_prob: np.ndarray, x_y: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """ROC-AUC debe superar el umbral mínimo bancario."""
        _, y_true = x_y
        auc = float(roc_auc_score(y_true, y_prob))
        assert auc >= self.AUC_MIN, f"AUC {auc:.4f} < umbral {self.AUC_MIN}"

    def test_gini_coefficient(
        self, y_prob: np.ndarray, x_y: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Coeficiente de Gini = 2 * AUC - 1."""
        _, y_true = x_y
        auc = float(roc_auc_score(y_true, y_prob))
        gini = 2 * auc - 1
        assert (
            gini >= self.GINI_MIN
        ), f"Gini {gini:.4f} < umbral {self.GINI_MIN}"

    def test_ks_statistic(
        self, y_prob: np.ndarray, x_y: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Distancia máxima entre buenos y malos (KS)."""
        _, y_true = x_y
        scores_bad = y_prob[y_true == 1]
        scores_good = y_prob[y_true == 0]
        ks_stat, _ = stats.ks_2samp(scores_bad, scores_good)
        assert (
            ks_stat >= self.KS_MIN
        ), f"KS {ks_stat:.4f} < umbral {self.KS_MIN}"


# ---------------------------------------------------------------------------
# 2. PRUEBAS DE ESTABILIDAD
# ---------------------------------------------------------------------------
class TestEstabilidad:
    """Valida estabilidad ante cambios de población."""

    PSI_MAX: float = 0.25

    def _calcular_psi(
        self,
        scores_base: np.ndarray,
        scores_actual: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Calcula el Population Stability Index."""
        breakpoints = np.percentile(scores_base, np.linspace(0, 100, bins + 1))
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf
        base_counts = np.histogram(scores_base, bins=breakpoints)[0]
        actual_counts = np.histogram(scores_actual, bins=breakpoints)[0]
        base_pct = np.maximum(base_counts / len(scores_base), 0.0001)
        actual_pct = np.maximum(actual_counts / len(scores_actual), 0.0001)
        psi = np.sum((actual_pct - base_pct) * np.log(actual_pct / base_pct))
        return float(psi)

    def test_psi_estabilidad(
        self, y_prob: np.ndarray, baseline_scores: np.ndarray
    ) -> None:
        """PSI entre scores actuales y baseline."""
        psi = self._calcular_psi(baseline_scores, y_prob)
        assert (
            psi <= self.PSI_MAX
        ), f"PSI {psi:.4f} > {self.PSI_MAX}. Población inestable."


# ---------------------------------------------------------------------------
# 3. PRUEBAS DE EXPLICABILIDAD (XAI)
# ---------------------------------------------------------------------------
class TestExplicabilidad:
    """Valida la lógica interna del modelo."""

    def test_shap_suma_a_prediccion(
        self, model: Any, x_y: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Propiedad de eficiencia SHAP."""
        x_data, _ = x_y
        x_sample = x_data[:10].astype(np.float64)
        explainer = shap.Explainer(model.predict_proba, x_sample)
        shap_values = explainer(x_sample)
        if len(shap_values.values.shape) == 3:
            sv = shap_values.values[:, :, 1]
            bv = shap_values.base_values[:, 1]
        else:
            sv = shap_values.values
            bv = shap_values.base_values
        y_pred = model.predict_proba(x_sample)[:, 1]
        shap_pred = sv.sum(axis=1) + bv
        np.testing.assert_allclose(shap_pred, y_pred, atol=1e-2)


# ---------------------------------------------------------------------------
# 4. PRUEBAS DE ÉTICA Y SESGO
# ---------------------------------------------------------------------------
class TestEticaSesgo:
    """Valida equidad en las decisiones del modelo."""

    THRESHOLD: float = 0.30
    DIR_MIN: float = 0.80

    def _disparate_impact_ratio(
        self, scores: np.ndarray, grupo_protegido: np.ndarray, threshold: float
    ) -> float:
        """Calcula el Disparate Impact Ratio."""
        tasa_ref = np.mean(scores[~grupo_protegido] < threshold)
        tasa_prot = np.mean(scores[grupo_protegido] < threshold)
        return float(tasa_prot / tasa_ref) if tasa_ref > 0 else 1.0

    def test_disparate_impact_sexo(
        self, model: Any, preprocessor: Any, dataset: pd.DataFrame
    ) -> None:
        """80% rule para género."""
        if "sex" not in dataset.columns:
            pytest.skip("Variable 'sex' no encontrada")
        x_data = preprocessor.transform(dataset.drop(columns=["risk"]))
        scores = model.predict_proba(x_data)[:, 1]
        es_mujer: np.ndarray = np.asarray(dataset["sex"] == "female")
        dir_ratio = self._disparate_impact_ratio(
            scores, es_mujer, self.THRESHOLD
        )
        assert (
            dir_ratio >= self.DIR_MIN
        ), f"DIR Sexo {dir_ratio:.4f} < {self.DIR_MIN}"
