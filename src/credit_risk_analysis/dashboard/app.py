from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pandas as pd
import shap
import streamlit as st
from dotenv import load_dotenv

from credit_risk_analysis.dashboard.services.model_service import (
    ModelService,
)


# ---------------------------------------------------------------------
# MATPLOTLIB DARK THEME
# ---------------------------------------------------------------------

plt.style.use("dark_background")

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "savefig.facecolor": "#0e1117",
    "axes.edgecolor": "#AAAAAA",
    "text.color": "#FAFAFA",
    "axes.labelcolor": "#FAFAFA",
    "xtick.color": "#CCCCCC",
    "ytick.color": "#CCCCCC",
})


# ---------------------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Analyzer Pro",
    page_icon="🛡️",
    layout="wide",
)

DEFAULT_THRESHOLD_VALUE = 0.387
SHAP_CONTRIBUTION_LABEL = "SHAP Contribution"

load_dotenv()

DEFAULT_THRESHOLD_ENV = os.getenv("DEFAULT_THRESHOLD")

if DEFAULT_THRESHOLD_ENV is None:
    st.warning(
        "⚠️ DEFAULT_THRESHOLD no encontrado. "
        f"Usando valor por defecto: {DEFAULT_THRESHOLD_VALUE}"
    )
    OFFICIAL_THRESHOLD = DEFAULT_THRESHOLD_VALUE
else:
    OFFICIAL_THRESHOLD = float(DEFAULT_THRESHOLD_ENV)


# ---------------------------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------------------------

@st.cache_resource
def init_model_service() -> ModelService:
    model_service = ModelService()
    model_service.initialize()
    return model_service


# ---------------------------------------------------------------------
# CSS DARK THEME
# ---------------------------------------------------------------------

DARK_CSS = """
<style>

[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #fafafa;
}

[data-testid="stSidebar"] {
    background-color: #1a1d23;
}

.stMetric {
    background-color: #1e2129;
    border-radius: 8px;
    padding: 10px;
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
}

</style>
"""


def apply_theme() -> None:
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# SHAP UTILITIES
# ---------------------------------------------------------------------

def _shorten_feature_names(
    names: List[str],
    max_len: int = 25,
) -> List[str]:

    shortened: List[str] = []

    for name in names:

        if len(name) > max_len:
            shortened.append(name[:max_len] + "…")
        else:
            shortened.append(name)

    return shortened


def _normalize_shap_values(
    shap_values: Any,
    feature_names: List[str],
) -> np.ndarray[Any, np.dtype[Any]]:

    arr = np.array(shap_values, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[0]

    if arr.ndim != 1:
        raise ValueError(
            f"No se pudo reducir shap_values a 1-D. Shape: {arr.shape}"
        )

    if len(arr) != len(feature_names):
        raise ValueError(
            f"Longitud SHAP ({len(arr)}) != Features "
            f"({len(feature_names)})"
        )

    return arr


def _normalize_feature_values(
    feature_values: Any,
    feature_names: List[str],
) -> np.ndarray[Any, np.dtype[Any]]:

    arr = np.array(feature_values, dtype=object)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[0]

    if len(arr) != len(feature_names):
        raise ValueError(
            f"Longitud values ({len(arr)}) != "
            f"features ({len(feature_names)})"
        )

    return arr


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

def render_sidebar() -> Tuple[
    Dict[str, Any],
    float,
    int,
    bool,
]:

    st.sidebar.header("📋 Parámetros Cliente")

    inputs: Dict[str, Any] = {
        "age": st.sidebar.slider("Edad", 18, 100, 30),
        "sex": st.sidebar.selectbox("Sexo", ["male", "female"]),
        "job": st.sidebar.selectbox("Trabajo", [0, 1, 2, 3], index=2),
        "housing": st.sidebar.selectbox(
            "Vivienda",
            ["own", "rent", "free"],
        ),
        "saving_accounts": st.sidebar.selectbox(
            "Ahorros",
            ["little", "moderate", "rich"],
        ),
        "checking_account": st.sidebar.selectbox(
            "C. Corriente",
            ["little", "moderate", "rich"],
        ),
        "credit_amount": st.sidebar.number_input(
            "Monto (USD)",
            value=5000,
        ),
        "duration": st.sidebar.slider(
            "Duración (meses)",
            1,
            72,
            24,
        ),
        "purpose": st.sidebar.selectbox(
            "Propósito",
            ["car", "business", "education", "repairs"],
        ),
    }

    st.sidebar.divider()

    st.sidebar.subheader("⚙️ Modelo")

    threshold = st.sidebar.slider(
        "Umbral de Decisión",
        0.0,
        1.0,
        OFFICIAL_THRESHOLD,
        0.001,
    )

    delta_m = st.sidebar.slider(
        "Simular variación monto",
        -2000,
        2000,
        0,
    )

    analyze = st.sidebar.button(
        "🚀 Analizar Escenario",
        use_container_width=True,
    )

    return inputs, threshold, delta_m, analyze


# ---------------------------------------------------------------------
# FORCE PLOT
# ---------------------------------------------------------------------

def plot_force(
    sv_1d: np.ndarray[Any, np.dtype[Any]],
    base_value: float,
    feature_names: List[str],
    fv_1d: np.ndarray[Any, np.dtype[Any]],
) -> Figure:

    plt.clf()

    feature_names = _shorten_feature_names(feature_names)

    shap.force_plot(
        base_value,
        sv_1d,
        fv_1d,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )

    fig = plt.gcf()

    fig.set_size_inches(16, 4)

    fig.patch.set_alpha(0)

    ax = plt.gca()

    ax.set_facecolor("#0e1117")

    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------
# WATERFALL PLOT
# ---------------------------------------------------------------------

def plot_waterfall(
    sv_1d: np.ndarray[Any, np.dtype[Any]],
    base_value: float,
    feature_names: List[str],
    fv_1d: np.ndarray[Any, np.dtype[Any]],
) -> Figure:

    plt.clf()

    feature_names = _shorten_feature_names(feature_names)

    explanation = shap.Explanation(
        values=sv_1d,
        base_values=float(base_value),
        data=fv_1d,
        feature_names=feature_names,
    )

    shap.plots.waterfall(
        explanation,
        max_display=8,
        show=False,
    )

    fig = plt.gcf()

    fig.set_size_inches(14, 6)

    fig.patch.set_alpha(0)

    ax = plt.gca()

    ax.set_facecolor("#0e1117")

    for artist in fig.findobj(match=Text):

        text = cast(Text, artist)

        text.set_color("#FAFAFA")

    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------
# SHAP EXPLANATIONS
# ---------------------------------------------------------------------

def display_shap_explanations(
    shap_values: Any,
    base_value: float,
    feature_names: List[str],
    feature_values: Any,
) -> None:

    st.divider()

    st.subheader("📊 Explicabilidad de la Decisión")

    try:

        sv_1d = _normalize_shap_values(
            shap_values,
            feature_names,
        )

        fv_1d = _normalize_feature_values(
            feature_values,
            feature_names,
        )

    except ValueError as exc:

        st.error(f"❌ Error al normalizar datos SHAP: {exc}")

        return

    st.markdown("### Force Plot: Contribución de Variables")

    try:

        f_fig = plot_force(
            sv_1d,
            base_value,
            feature_names,
            fv_1d,
        )

        st.pyplot(
            f_fig,
            use_container_width=True,
        )

        plt.close(f_fig)

    except Exception as exc:

        st.warning(
            f"No se pudo generar Force Plot: {exc}"
        )

    st.markdown("### Waterfall: Magnitud del Impacto")

    try:

        w_fig = plot_waterfall(
            sv_1d,
            base_value,
            feature_names,
            fv_1d,
        )

        st.pyplot(
            w_fig,
            use_container_width=True,
        )

        plt.close(w_fig)

    except Exception as exc:

        st.warning(
            f"No se pudo generar Waterfall Plot: {exc}"
        )

    with st.expander("📋 Ver contribuciones detalladas"):

        contributions = pd.DataFrame({
            "Feature": feature_names,
            "Value": fv_1d.tolist(),
            SHAP_CONTRIBUTION_LABEL: sv_1d.tolist(),
        })

        col = SHAP_CONTRIBUTION_LABEL

        contributions["Impacto"] = contributions[col].apply(
            lambda x: (
                "Aumenta Riesgo"
                if x > 0
                else "Reduce Riesgo"
            )
        )

        contributions["Magnitud"] = contributions[col].abs()

        contributions = contributions.sort_values(
            "Magnitud",
            ascending=False,
        )

        st.dataframe(
            contributions,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# RESULT DISPLAY
# ---------------------------------------------------------------------

def display_prediction_results(
    result: Dict[str, Any],
    threshold: float,
) -> None:

    prob = result["probability"]

    feature_names = result.get(
        "feature_names",
        [],
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Score de Riesgo",
        f"{prob:.2%}",
    )

    col2.metric(
        "Umbral Aplicado",
        f"{threshold:.3f}",
    )

    is_high = prob >= threshold

    status = (
        "RIESGO ALTO"
        if is_high
        else "RIESGO BAJO"
    )

    col3.metric(
        "Evaluación",
        status,
    )

    if is_high:

        st.error(
            f"⚠️ {status}: {prob:.2%} >= {threshold:.2%}"
        )

    else:

        st.success(
            f"✅ {status}: {prob:.2%} < {threshold:.2%}"
        )

    if (
        result.get("shap_values") is not None
        and feature_names
    ):

        display_shap_explanations(
            result["shap_values"],
            result["base_value"],
            feature_names,
            result["feature_values"],
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    model_service = init_model_service()

    inputs, threshold, delta_m, analyze = render_sidebar()

    apply_theme()

    st.title("🛡️ Credit Risk Analyzer Pro")

    st.caption(
        f"Configuración Activa: Threshold @ {threshold:.3f}"
    )

    if analyze:

        if delta_m != 0:

            inputs["credit_amount"] += delta_m

            st.info(
                f"💰 Monto simulado: "
                f"${inputs['credit_amount']:,} USD "
                f"(variación: {delta_m:+d} USD)"
            )

        result = model_service.predict(inputs)

        if result.get("status") == "success":

            display_prediction_results(
                result,
                threshold,
            )

        else:

            st.error(
                f"❌ Error en predicción: "
                f"{result.get('message')}"
            )

    else:

        st.info(
            "Ajuste los parámetros y "
            "presione Analizar Escenario."
        )

        with st.expander(
            "ℹ️ ¿Qué es SHAP?"
        ):

            st.markdown(
                """
**SHAP (SHapley Additive Explanations)** explica la
predicción del modelo.

- **Force Plot** muestra cómo cada variable empuja
  el resultado.

- **Waterfall Plot** desglosa el impacto paso a paso.

- **Base Value** representa el valor promedio del
  modelo.
"""
            )


if __name__ == "__main__":
    main()
