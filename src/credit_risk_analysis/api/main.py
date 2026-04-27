from __future__ import annotations

import os
import time
import functools
from typing import Any, Dict, Callable, TypeVar, cast, Union, List, Optional

import mlflow
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from credit_risk_analysis.dashboard.services.model_service import ModelService
from credit_risk_analysis.utils.logging_config import setup_logger
from credit_risk_analysis.api.schemas import (
    CreditFeatures,
    BatchPredictRequest,
)

F = TypeVar("F", bound=Callable[..., Any])
logger = setup_logger(name="api_main")

# --- CONFIGURACIÓN ---
MLFLOW_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "http://mlflow_service:5000"
)
mlflow.set_tracking_uri(MLFLOW_URI)
RUN_ID: str = os.getenv("RUN_ID", "")
DEFAULT_THRESHOLD: float = float(os.getenv("DEFAULT_THRESHOLD", "0.387"))
MODEL_UNAVAILABLE: str = "Modelo no disponible. Intenta en unos segundos."

# --- MÉTRICAS PROMETHEUS ---
REQ_CTR = Counter(
    "http_req_total", "Total de peticiones", ["endpoint", "status"]
)
LAT_HIST = Histogram(
    "pred_latency_seconds", "Latencia por endpoint", ["endpoint"]
)
RISK_GAUGE = Gauge("last_risk_score", "Score de la última predicción")
AUC_GAUGE = Gauge("model_auc_score", "AUC actual de MLflow")
KS_GAUGE = Gauge("model_ks_score", "KS actual de MLflow")
THRESH_GAUGE = Gauge("model_threshold", "Umbral de decisión activo")
DECISION_CTR = Counter("decisions_total", "Conteo decisiones", ["decision"])
HIGH_RISK_CTR = Counter("high_risk_total", "Total acumulado alto riesgo")
LOW_RISK_CTR = Counter("low_risk_total", "Total acumulado bajo riesgo")

BATCH_SIZE_HIST = Histogram(
    "batch_size", "Tamaño lotes", buckets=[1, 5, 10, 20, 50, 100]
)
BATCH_TIME = Histogram(
    "batch_duration", "Tiempo batch", buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)
BATCH_AVG_GAUGE = Gauge("batch_avg_risk", "Promedio riesgo último lote")


def track_telemetry(endpoint: str) -> Callable[[F], F]:
    """Decorador para métricas de Prometheus y trazabilidad."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_perf = time.perf_counter()
            try:
                res = await func(*args, **kwargs)
                latency = time.perf_counter() - start_perf
                LAT_HIST.labels(endpoint=endpoint).observe(latency)
                REQ_CTR.labels(endpoint=endpoint, status="success").inc()
                return res
            except Exception as exc:
                REQ_CTR.labels(endpoint=endpoint, status="error").inc()
                logger.error("Fallo en [%s]: %s", endpoint, exc)
                if isinstance(exc, HTTPException):
                    raise exc
                raise HTTPException(status_code=500, detail="Internal Error")
        return cast(F, wrapper)
    return decorator


def _process_profile(
    idx: int,
    profile: Any,
    stats: Dict[str, int]
) -> Optional[Dict[str, Any]]:
    """Procesa un perfil individual para reducir complejidad."""
    try:
        prediction = model_service.predict(profile.model_dump())
        if prediction.get("status") == "error":
            stats["error_count"] += 1
            return None

        prob = float(prediction.get("probability", 0.0))
        dec = str(prediction.get("decision", "unknown"))

        DECISION_CTR.labels(decision=dec).inc()
        if dec == "high_risk":
            stats["high_risk_count"] += 1
        else:
            stats["low_risk_count"] += 1

        return {"index": idx, "score": round(prob, 4), "decision": dec}
    except Exception as err:
        logger.error("🚨 Error en registro %d: %s", idx, str(err))
        stats["error_count"] += 1
        return None


# --- APP FASTAPI ---
_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    422: {"description": "Error de validación Pydantic"},
    500: {"description": "Fallo en lógica interna"},
    503: {"description": "Modelo no cargado"},
}

app = FastAPI(
    title="Credit Risk ML API",
    description="API de Producción - Riesgo Crediticio",
    version="1.4.0",
)

model_service = ModelService()


def sync_mlflow_metrics() -> None:
    """Sincroniza métricas desde MLflow."""
    if not RUN_ID:
        return
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(RUN_ID).data
        m = run.metrics
        auc_v = m.get("training_roc_auc") or m.get("auc") or 0.0
        ks_v = m.get("training_ks") or m.get("ks_score") or 0.0
        AUC_GAUGE.set(float(auc_v))
        KS_GAUGE.set(float(ks_v))
        raw_t = run.params.get("threshold")
        THRESH_GAUGE.set(float(raw_t) if raw_t else DEFAULT_THRESHOLD)
        logger.info("✅ Gobernanza OK: AUC=%.4f", auc_v)
    except Exception as exc:
        logger.error("❌ Fallo MLflow: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    """Inicia el servicio y carga el modelo."""
    sync_mlflow_metrics()
    try:
        # Usamos initialize() del ModelService refactorizado
        model_service.initialize()
    except Exception as exc:
        logger.critical("🚨 Fallo crítico: %s", exc)


@app.get("/health", responses=_RESPONSES, tags=["Sistema"])
@track_telemetry(endpoint="/health")
async def health_check() -> Dict[str, Any]:
    """Valida carga de modelo y preprocesador."""
    m_ok = model_service.is_ready
    return {
        "status": "online" if m_ok else "degraded",
        "model_loaded": m_ok,
        "run_id": RUN_ID
    }


@app.post("/predict", responses=_RESPONSES, tags=["Inferencia"])
@track_telemetry(endpoint="/predict")
async def predict_single(features: CreditFeatures) -> Dict[str, Any]:
    """Inferencia unitaria con validación Pydantic."""
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE)
    res = model_service.predict(features.model_dump())
    if res.get("status") == "error":
        raise HTTPException(status_code=500, detail=str(res.get("message")))
    prob = float(res.get("probability", 0.0))
    dec = str(res.get("decision", "unknown"))
    RISK_GAUGE.set(prob)
    DECISION_CTR.labels(decision=dec).inc()
    if dec == "high_risk":
        HIGH_RISK_CTR.inc()
    else:
        LOW_RISK_CTR.inc()
    return res


@app.post("/predict-batch", responses=_RESPONSES, tags=["Inferencia"])
@track_telemetry(endpoint="/predict-batch")
async def predict_batch_process(
    request: BatchPredictRequest
) -> Dict[str, Any]:
    """Motor de Inferencia Masiva."""
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE)

    start_perf = time.perf_counter()
    results: List[Dict[str, Any]] = []
    probs: List[float] = []
    stats = {"high_risk_count": 0, "low_risk_count": 0, "error_count": 0}

    for idx, profile in enumerate(request.profiles):
        res = _process_profile(idx, profile, stats)
        if res:
            results.append(res)
            probs.append(res["score"])

    valid = len(probs)
    duration = time.perf_counter() - start_perf
    avg_risk = sum(probs) / valid if valid > 0 else 0.0

    BATCH_SIZE_HIST.observe(len(request.profiles))
    BATCH_TIME.observe(duration)
    BATCH_AVG_GAUGE.set(avg_risk)
    HIGH_RISK_CTR.inc(stats["high_risk_count"])
    LOW_RISK_CTR.inc(stats["low_risk_count"])

    return {
        "status": "success" if stats["error_count"] == 0 else "partial",
        "execution_summary": {
            "requested": len(request.profiles),
            "processed": valid,
            "errors": stats["error_count"],
            "seconds": round(duration, 4)
        },
        "analytics": {
            "avg_score": round(avg_risk, 4),
            "max_score": round(max(probs) if valid else 0.0, 4),
            "risk_ratio": round(stats["high_risk_count"] / valid, 2)
            if valid > 0 else 0.0
        },
        "predictions": results
    }


@app.get("/metrics", tags=["Observabilidad"])
def get_metrics() -> Response:
    """Expone métricas para Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
