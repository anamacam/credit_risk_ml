"""Dashboard de monitoreo del modelo de riesgo crediticio."""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any, Dict

import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Risk Monitor", page_icon="📊", layout="wide"
)

MLFLOW_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "http://mlflow_service:5000"
)
API_URL: str = os.getenv("API_URL", "http://risk_api:8000")
PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", "/app"))
DB_PATH: Path = PROJECT_ROOT / "logs" / "predictions.db"
AUDIT_PATH: Path = PROJECT_ROOT / "logs" / "audit_log.jsonl"

mlflow.set_tracking_uri(MLFLOW_URI)


@st.cache_data(ttl=30)
def load_predictions() -> pd.DataFrame:
    """Carga predicciones desde SQLite."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df: pd.DataFrame = pd.read_sql_query(
                "SELECT * FROM model_logs", conn
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def load_audit_log() -> pd.DataFrame:
    """Carga el log de auditoria JSONL."""
    if not AUDIT_PATH.exists():
        return pd.DataFrame()
    try:
        records = []
        with open(AUDIT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_mlflow_runs() -> pd.DataFrame:
    """Carga los runs desde MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=20,
            )
            for run in runs:
                all_runs.append({
                    "run_id": run.info.run_id[:8],
                    "experiment": exp.name,
                    "status": run.info.status,
                    "recall": run.data.metrics.get("recall"),
                    "f1_score": run.data.metrics.get("f1_score"),
                    "precision": run.data.metrics.get("precision"),
                    "roc_auc": run.data.metrics.get("roc_auc"),
                    "threshold": run.data.params.get("threshold"),
                })
        return pd.DataFrame(all_runs) if all_runs else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def check_api_health() -> Dict[str, Any]:
    """Verifica el estado de la API."""
    result: Dict[str, Any] = {
        "status": "error", "detail": "unknown"
    }
    try:
        with urllib.request.urlopen(
            f"{API_URL}/health", timeout=5
        ) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        result = {"status": "error", "detail": str(exc)}
    return result


st.title("📊 Monitor de Modelo — Credit Risk")
st.caption(f"MLflow: `{MLFLOW_URI}` · API: `{API_URL}`")

col1, col2, col3, col4 = st.columns(4)
health = check_api_health()
api_ok = health.get("status") == "ok"
model_ok = bool(health.get("model_loaded", False))
prep_ok = bool(health.get("preprocessor_loaded", False))
col1.metric("API", "✅ Online" if api_ok else "❌ Offline")
col2.metric("Modelo", "✅ Cargado" if model_ok else "❌ No cargado")
col3.metric(
    "Preprocesador", "✅ Cargado" if prep_ok else "❌ No cargado"
)
col4.metric("Run ID", health.get("run_id") or "N/A")
st.divider()

st.subheader("Experimentos MLflow")
runs_df = load_mlflow_runs()
if runs_df.empty:
    st.info("No hay runs registrados en MLflow todavia.")
else:
    st.dataframe(runs_df, use_container_width=True, hide_index=True)
    numeric_cols = ["recall", "f1_score", "precision", "roc_auc"]
    available = [c for c in numeric_cols if c in runs_df.columns]
    if available:
        best = runs_df[available].max()
        mcols = st.columns(len(available))
        for i, col_name in enumerate(available):
            val = best[col_name]
            mcols[i].metric(
                col_name.upper(),
                f"{val:.4f}" if pd.notna(val) else "N/A",
            )
st.divider()

st.subheader("Predicciones en produccion")
pred_df = load_predictions()
if pred_df.empty:
    st.info("No hay predicciones registradas aun.")
else:
    total = len(pred_df)
    if "probability" in pred_df.columns:
        avg_prob = float(pred_df["probability"].mean())
        high_risk = int((pred_df["probability"] >= 0.30).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predicciones", total)
        c2.metric("Probabilidad media", f"{avg_prob:.4f}")
        c3.metric(
            "Alto riesgo (>=0.30)",
            f"{high_risk} ({100*high_risk/total:.1f}%)",
        )
        if "timestamp" in pred_df.columns:
            st.line_chart(
                pred_df.set_index("timestamp")["probability"],
                use_container_width=True,
            )
        else:
            st.line_chart(
                pred_df["probability"], use_container_width=True
            )
    st.dataframe(
        pred_df.tail(50), use_container_width=True, hide_index=True
    )
st.divider()

st.subheader("Log de auditoria")
audit_df = load_audit_log()
if audit_df.empty:
    st.info("No hay entradas de auditoria todavia.")
else:
    st.metric("Total entradas auditadas", len(audit_df))
    st.dataframe(
        audit_df.tail(20), use_container_width=True, hide_index=True
    )

with st.sidebar:
    st.header("Controles")
    if st.button("Refrescar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown(f"[MLflow UI]({MLFLOW_URI})")
    st.markdown(f"[API Docs]({API_URL}/docs)")
    st.markdown("[Grafana](http://localhost:3000)")
    st.markdown("[Prometheus](http://localhost:9090)")
    st.markdown("[Prometheus](http://localhost:9090)")
