"""
Suite de Stress Testing y Bias Testing para el modelo de riesgo crediticio.

Exporta metricas a Prometheus pushgateway para visualizacion en Grafana.
Si no hay pushgateway disponible, genera reporte HTML local.

Uso:
    python stress_bias_testing.py --mode stress
    python stress_bias_testing.py --mode bias
    python stress_bias_testing.py --mode all
    python stress_bias_testing.py --mode all \
        --pushgateway http://localhost:9091
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
API_URL: str = os.getenv("API_URL", "http://localhost:8000")
PREDICT_URL: str = f"{API_URL}/predict"
METRICS_URL: str = f"{API_URL}/metrics"
PUSHGATEWAY: str = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
REPORT_PATH: Path = Path("stress_bias_report.html")

BASE_PROFILE: dict[str, Any] = {
    "age": 35,
    "sex": "male",
    "job": 2,
    "housing": "own",
    "saving_accounts": "moderate",
    "checking_account": "moderate",
    "credit_amount": 5000,
    "duration": 24,
    "purpose": "car",
}


# ---------------------------------------------------------------------------
# Dataclasses de resultados
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    """Resultado de una peticion individual."""
    latency_ms: float
    status_code: int
    probability: float | None
    decision: str | None
    error: str | None


@dataclass
class StressResult:
    """Resultado agregado de una prueba de stress."""
    test_name: str
    total_requests: int
    success_count: int
    error_count: int
    latencies: list[float] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def avg_probability(self) -> float:
        return (
            statistics.mean(self.probabilities)
            if self.probabilities else 0.0
        )


@dataclass
class BiasResult:
    """Resultado de analisis de sesgo para un grupo."""
    group_name: str
    group_value: str
    total: int
    high_risk_count: int
    probabilities: list[float] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.high_risk_count / self.total

    @property
    def avg_probability(self) -> float:
        return (
            statistics.mean(self.probabilities)
            if self.probabilities else 0.0
        )


# ---------------------------------------------------------------------------
# Helper de peticion
# ---------------------------------------------------------------------------
def send_request(payload: dict[str, Any]) -> RequestResult:
    """Envia una prediccion a la API y mide la latencia."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            PREDICT_URL,
            json=payload,
            timeout=10,
        )
        latency = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            body = resp.json()
            return RequestResult(
                latency_ms=latency,
                status_code=resp.status_code,
                probability=body.get("probability"),
                decision=body.get("decision"),
                error=None,
            )
        return RequestResult(
            latency_ms=latency,
            status_code=resp.status_code,
            probability=None,
            decision=None,
            error=f"HTTP {resp.status_code}",
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(
            latency_ms=latency,
            status_code=0,
            probability=None,
            decision=None,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# 1. STRESS TESTING
# ---------------------------------------------------------------------------
class StressTester:
    """Ejecuta pruebas de stress sobre la API."""

    def run_volume_test(
        self,
        n_requests: int = 100,
        workers: int = 20,
    ) -> StressResult:
        """
        Volumen y latencia: rafaga de peticiones simultaneas.
        Mide degradacion de latencia bajo carga.
        """
        print(
            f"\n[STRESS] Volumen: {n_requests} requests "
            f"con {workers} workers..."
        )
        result = StressResult(
            test_name="volume_stress",
            total_requests=n_requests,
            success_count=0,
            error_count=0,
        )

        payloads = [BASE_PROFILE.copy() for _ in range(n_requests)]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = [
                executor.submit(send_request, p) for p in payloads
            ]
            for future in concurrent.futures.as_completed(futures):
                r = future.result()
                result.latencies.append(r.latency_ms)
                if r.error is None and r.probability is not None:
                    result.success_count += 1
                    result.probabilities.append(r.probability)
                else:
                    result.error_count += 1

        print(
            f"  Exito: {result.success_rate:.1%} | "
            f"P50: {result.p50:.0f}ms | "
            f"P95: {result.p95:.0f}ms | "
            f"P99: {result.p99:.0f}ms"
        )
        return result

    def run_corruption_test(self) -> StressResult:
        """
        Corrupcion de datos: valores extremos, nulos, tipos incorrectos.
        El modelo debe responder con error controlado, no crashear.
        """
        print("\n[STRESS] Corrupcion de datos...")

        rng = np.random.default_rng(42)
        corrupt_payloads: list[dict[str, Any]] = [
            # Valores extremos numericos
            {**BASE_PROFILE, "age": 0},
            {**BASE_PROFILE, "age": 999},
            {**BASE_PROFILE, "age": -50},
            {**BASE_PROFILE, "credit_amount": 0},
            {**BASE_PROFILE, "credit_amount": 999_999_999},
            {**BASE_PROFILE, "credit_amount": -1000},
            {**BASE_PROFILE, "duration": 0},
            {**BASE_PROFILE, "duration": 9999},
            # Valores nulos
            {**BASE_PROFILE, "age": None},
            {**BASE_PROFILE, "sex": None},
            {**BASE_PROFILE, "credit_amount": None},
            # Ruido gaussiano intenso
            {
                **BASE_PROFILE,
                "age": int(rng.normal(35, 1000)),
                "credit_amount": max(
                    1, int(rng.normal(5000, 50000))
                ),
                "duration": max(1, int(rng.normal(24, 200))),
            },
            # Categorias invalidas
            {**BASE_PROFILE, "sex": "unknown_gender"},
            {**BASE_PROFILE, "housing": "spaceship"},
            {**BASE_PROFILE, "purpose": "drug_trafficking"},
            # Tipos incorrectos
            {**BASE_PROFILE, "age": "treinta"},
            {**BASE_PROFILE, "job": "manager"},
            {**BASE_PROFILE, "credit_amount": "mucho"},
            # Payload vacio
            {},
            # Todos nulos
            dict.fromkeys(BASE_PROFILE),
        ]

        result = StressResult(
            test_name="corruption_stress",
            total_requests=len(corrupt_payloads),
            success_count=0,
            error_count=0,
        )

        for payload in corrupt_payloads:
            r = send_request(payload)
            result.latencies.append(r.latency_ms)
            if r.status_code != 503:
                result.success_count += 1
            else:
                result.error_count += 1
            if r.probability is not None:
                result.probabilities.append(r.probability)

        print(
            f"  API no crasheo en: "
            f"{result.success_rate:.1%} de casos corruptos"
        )
        return result

    def run_drift_simulation(self) -> StressResult:
        """
        Deriva de datos: simula datos postpandemia con
        distribucion muy diferente al entrenamiento.
        Crisis: sin ahorros, creditos altos, duraciones maximas.
        """
        print("\n[STRESS] Simulacion de data drift (escenario crisis)...")

        rng = np.random.default_rng(42)
        crisis_choices = ["business", "repairs"]
        sex_choices = ["male", "female"]
        crisis_profiles = []
        for _ in range(50):
            profile = {
                "age": int(rng.choice([22, 23, 24, 25])),
                "sex": str(rng.choice(sex_choices)),
                "job": 0,
                "housing": "rent",
                "saving_accounts": "little",
                "checking_account": "little",
                "credit_amount": int(rng.uniform(10000, 20000)),
                "duration": int(rng.choice([60, 66, 72])),
                "purpose": str(rng.choice(crisis_choices)),
            }
            crisis_profiles.append(profile)

        normal_purposes = ["car", "furniture/equipment"]
        normal_profiles = []
        for _ in range(50):
            profile = {
                "age": int(rng.uniform(30, 55)),
                "sex": str(rng.choice(sex_choices)),
                "job": int(rng.choice([1, 2, 3])),
                "housing": str(rng.choice(["own", "own", "rent"])),
                "saving_accounts": str(
                    rng.choice(["moderate", "rich"])
                ),
                "checking_account": str(
                    rng.choice(["moderate", "rich"])
                ),
                "credit_amount": int(rng.uniform(2000, 8000)),
                "duration": int(rng.uniform(12, 36)),
                "purpose": str(rng.choice(normal_purposes)),
            }
            normal_profiles.append(profile)

        result_normal = StressResult(
            test_name="drift_normal",
            total_requests=50,
            success_count=0,
            error_count=0,
        )
        result_crisis = StressResult(
            test_name="drift_crisis",
            total_requests=50,
            success_count=0,
            error_count=0,
        )

        for p in normal_profiles:
            r = send_request(p)
            if r.probability is not None:
                result_normal.success_count += 1
                result_normal.probabilities.append(r.probability)
                result_normal.latencies.append(r.latency_ms)
            else:
                result_normal.error_count += 1

        for p in crisis_profiles:
            r = send_request(p)
            if r.probability is not None:
                result_crisis.success_count += 1
                result_crisis.probabilities.append(r.probability)
                result_crisis.latencies.append(r.latency_ms)
            else:
                result_crisis.error_count += 1

        drift_delta = (
            result_crisis.avg_probability
            - result_normal.avg_probability
        )
        print(
            f"  Score normal: {result_normal.avg_probability:.4f} | "
            f"Score crisis: {result_crisis.avg_probability:.4f} | "
            f"Delta drift: {drift_delta:+.4f}"
        )

        # Retornar el resultado de crisis para las metricas
        result_crisis.test_name = "drift_simulation"
        return result_crisis

    def run_dos_simulation(
        self,
        n_requests: int = 200,
        workers: int = 50,
    ) -> StressResult:
        """
        Simulacion de DoS: maxima carga simultanea.
        Mide si la API mantiene disponibilidad bajo ataque.
        """
        print(
            f"\n[STRESS] DoS simulation: "
            f"{n_requests} requests con {workers} workers..."
        )
        result = StressResult(
            test_name="dos_simulation",
            total_requests=n_requests,
            success_count=0,
            error_count=0,
        )

        payloads = []
        for _ in range(n_requests):
            p = BASE_PROFILE.copy()
            # Variaciones para evitar cache
            rng = np.random.default_rng(42)
            p["credit_amount"] = int(rng.uniform(1000, 20000))
            p["duration"] = int(rng.choice(range(6, 73, 6)))
            payloads.append(p)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = [
                executor.submit(send_request, p) for p in payloads
            ]
            for future in concurrent.futures.as_completed(futures):
                r = future.result()
                result.latencies.append(r.latency_ms)
                if r.error is None and r.probability is not None:
                    result.success_count += 1
                    result.probabilities.append(r.probability)
                else:
                    result.error_count += 1

        print(
            f"  Disponibilidad: {result.success_rate:.1%} | "
            f"P99: {result.p99:.0f}ms"
        )
        return result


# ---------------------------------------------------------------------------
# 2. BIAS TESTING
# ---------------------------------------------------------------------------
class BiasTester:
    """Pruebas de sesgo y equidad del modelo."""

    THRESHOLD = 0.30

    def run_subgroup_analysis(self) -> list[BiasResult]:
        """
        Slicing: compara tasa de rechazo por grupo.
        Detecta si ciertos grupos son rechazados sistematicamente.
        """
        print("\n[BIAS] Analisis de subgrupos (slicing)...")

        groups: dict[str, list[Any]] = {
            "sex": ["male", "female"],
            "job": [0, 1, 2, 3],
            "housing": ["own", "rent", "free"],
        }

        results = []

        for group_name, group_values in groups.items():
            for value in group_values:
                bias_result = BiasResult(
                    group_name=group_name,
                    group_value=str(value),
                    total=0,
                    high_risk_count=0,
                )

                # 30 peticiones por subgrupo
                for age in range(25, 55, 1):
                    profile = {**BASE_PROFILE, group_name: value}
                    profile["age"] = age
                    r = send_request(profile)
                    if r.probability is not None:
                        bias_result.total += 1
                        bias_result.probabilities.append(
                            r.probability
                        )
                        if r.decision == "high_risk":
                            bias_result.high_risk_count += 1

                results.append(bias_result)
                print(
                    f"  {group_name}={value}: "
                    f"rechazo={bias_result.rejection_rate:.1%} | "
                    f"prob_media={bias_result.avg_probability:.4f}"
                )

        return results

    def run_counterfactual_test(self) -> list[dict[str, Any]]:
        """
        Contrafactuales: cambia una sola variable sensible
        y mide el cambio en la prediccion.
        Si cambia solo por sexo o edad, hay sesgo.
        """
        print("\n[BIAS] Pruebas contrafactuales...")

        contrafactuales: list[dict[str, Any]] = [
            {
                "descripcion": "Hombre vs Mujer — mismo perfil",
                "variable": "sex",
                "valor_a": "male",
                "valor_b": "female",
            },
            {
                "descripcion": "Trabajo calificado vs no calificado",
                "variable": "job",
                "valor_a": 3,
                "valor_b": 0,
            },
            {
                "descripcion": "Vivienda propia vs alquiler",
                "variable": "housing",
                "valor_a": "own",
                "valor_b": "rent",
            },
            {
                "descripcion": "Ahorros altos vs bajos",
                "variable": "saving_accounts",
                "valor_a": "rich",
                "valor_b": "little",
            },
        ]

        results = []

        for test in contrafactuales:
            profile_a = {
                **BASE_PROFILE,
                test["variable"]: test["valor_a"],
            }
            profile_b = {
                **BASE_PROFILE,
                test["variable"]: test["valor_b"],
            }

            r_a = send_request(profile_a)
            r_b = send_request(profile_b)

            if r_a.probability is not None and r_b.probability is not None:
                delta = r_b.probability - r_a.probability
                decision_changed = r_a.decision != r_b.decision
                result = {
                    "test": test["descripcion"],
                    "variable": test["variable"],
                    "prob_a": round(r_a.probability, 4),
                    "prob_b": round(r_b.probability, 4),
                    "delta": round(delta, 4),
                    "decision_cambio": decision_changed,
                    "sesgo_detectado": (
                        abs(delta) > 0.10 and decision_changed
                    ),
                }
                results.append(result)
                sesgo = "⚠️ SESGO" if result["sesgo_detectado"] else "✅ OK"
                print(
                    f"  {test['descripcion']}: "
                    f"delta={delta:+.4f} | "
                    f"decision_cambio={decision_changed} | {sesgo}"
                )

        return results

    def run_fairness_metrics(
        self,
        subgroup_results: list[BiasResult],
    ) -> dict[str, Any]:
        """
        Metricas de equidad:
        - Paridad demografica: diferencia max en tasa de rechazo
        - Disparate Impact Ratio (DIR)
        - Equal Opportunity gap
        """
        print("\n[BIAS] Metricas de equidad (fairness)...")

        fairness: dict[str, Any] = {}

        for group_name in ["sex", "job", "housing"]:
            group_data = [
                r for r in subgroup_results
                if r.group_name == group_name
            ]
            if not group_data:
                continue

            rejection_rates = [r.rejection_rate for r in group_data]
            max_rate = max(rejection_rates)
            min_rate = min(rejection_rates)
            paridad_gap = max_rate - min_rate

            # Disparate Impact Ratio
            if max_rate > 0:
                dir_ratio = min_rate / max_rate
            else:
                dir_ratio = 1.0

            fairness[group_name] = {
                "paridad_demografica_gap": round(paridad_gap, 4),
                "disparate_impact_ratio": round(dir_ratio, 4),
                "cumple_80_rule": dir_ratio >= 0.80,
                "max_rechazo": round(max_rate, 4),
                "min_rechazo": round(min_rate, 4),
            }

            cumple = "✅" if dir_ratio >= 0.80 else "⚠️ VIOLA 80% RULE"
            print(
                f"  {group_name}: "
                f"DIR={dir_ratio:.4f} | "
                f"gap={paridad_gap:.4f} | {cumple}"
            )

        return fairness


# ---------------------------------------------------------------------------
# Exportacion a Prometheus Pushgateway
# ---------------------------------------------------------------------------
def push_metrics_to_prometheus(
    stress_results: list[StressResult],
    bias_results: list[BiasResult],
    fairness: dict[str, Any],
    pushgateway_url: str,
) -> None:
    """
    Exporta resultados al Prometheus Pushgateway para
    visualizacion en Grafana.
    """
    lines = []

    # Metricas de stress
    for r in stress_results:
        test = r.test_name
        lines.append(
            f'stress_success_rate{{test="{test}"}} {r.success_rate}'
        )
        lines.append(
            f'stress_latency_p50_ms{{test="{test}"}} {r.p50}'
        )
        lines.append(
            f'stress_latency_p95_ms{{test="{test}"}} {r.p95}'
        )
        lines.append(
            f'stress_latency_p99_ms{{test="{test}"}} {r.p99}'
        )
        lines.append(
            f'stress_avg_probability{{test="{test}"}} {r.avg_probability}'
        )
        lines.append(
            f'stress_error_count{{test="{test}"}} {r.error_count}'
        )

    # Metricas de bias
    for b in bias_results:
        lines.append(
            f'bias_rejection_rate{{'
            f'group="{b.group_name}",'
            f'value="{b.group_value}"'
            f'}} {b.rejection_rate}'
        )
        lines.append(
            f'bias_avg_probability{{'
            f'group="{b.group_name}",'
            f'value="{b.group_value}"'
            f'}} {b.avg_probability}'
        )

    # Metricas de fairness
    for group, metrics in fairness.items():
        lines.append(
            f'fairness_dir_ratio{{group="{group}"}} '
            f'{metrics["disparate_impact_ratio"]}'
        )
        lines.append(
            f'fairness_paridad_gap{{group="{group}"}} '
            f'{metrics["paridad_demografica_gap"]}'
        )
        lines.append(
            f'fairness_cumple_80rule{{group="{group}"}} '
            f'{1 if metrics["cumple_80_rule"] else 0}'
        )

    payload = "\n".join(lines) + "\n"

    try:
        resp = requests.post(
            f"{pushgateway_url}/metrics/job/credit_risk_testing",
            data=payload,
            headers={"Content-Type": "text/plain"},
            timeout=5,
        )
        if resp.status_code in [200, 202]:
            print(
                f"\n✅ Metricas enviadas a Pushgateway: {pushgateway_url}"
            )
        else:
            print(f"\n⚠️ Pushgateway retorno {resp.status_code}")
    except Exception as exc:
        print(f"\n⚠️ No se pudo conectar al Pushgateway: {exc}")
        print("  Las metricas se guardaron en el reporte HTML.")


# ---------------------------------------------------------------------------
# Reporte HTML
# ---------------------------------------------------------------------------
def generate_html_report(
    stress_results: list[StressResult],
    bias_results: list[BiasResult],
    counterfactual_results: list[dict[str, Any]],
    fairness: dict[str, Any],
) -> None:
    """Genera reporte HTML con graficas interactivas via Chart.js."""

    stress_labels = json.dumps([r.test_name for r in stress_results])
    stress_p50 = json.dumps([round(r.p50, 1) for r in stress_results])
    stress_p95 = json.dumps([round(r.p95, 1) for r in stress_results])
    stress_p99 = json.dumps([round(r.p99, 1) for r in stress_results])
    stress_success = json.dumps(
        [round(r.success_rate * 100, 1) for r in stress_results]
    )

    sex_data = [
        b for b in bias_results if b.group_name == "sex"
    ]
    sex_labels = json.dumps([b.group_value for b in sex_data])
    sex_rejection = json.dumps(
        [round(b.rejection_rate * 100, 2) for b in sex_data]
    )

    job_data = [b for b in bias_results if b.group_name == "job"]
    job_labels = json.dumps([f"Job {b.group_value}" for b in job_data])
    job_rejection = json.dumps(
        [round(b.rejection_rate * 100, 2) for b in job_data]
    )

    cf_labels = json.dumps(
        [r["variable"] for r in counterfactual_results]
    )
    cf_delta = json.dumps(
        [round(r["delta"] * 100, 2) for r in counterfactual_results]
    )

    dir_labels = json.dumps(list(fairness.keys()))
    dir_values = json.dumps(
        [v["disparate_impact_ratio"] for v in fairness.values()]
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Stress & Bias Testing Report — Credit Risk</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: Arial, sans-serif; background: #0f1117;
         color: #e0e0e0; margin: 0; padding: 24px; }}
  h1 {{ color: #4fc3f7; border-bottom: 2px solid #4fc3f7;
       padding-bottom: 8px; }}
  h2 {{ color: #81d4fa; margin-top: 40px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr;
           gap: 24px; margin-top: 16px; }}
  .card {{ background: #1e2130; border-radius: 12px;
           padding: 20px; }}
  .card h3 {{ margin-top: 0; color: #b3e5fc; }}
  .metric {{ display: inline-block; background: #263045;
             border-radius: 8px; padding: 10px 18px;
             margin: 6px; text-align: center; }}
  .metric .val {{ font-size: 1.6em; font-weight: bold;
                  color: #4fc3f7; }}
  .metric .lbl {{ font-size: 0.8em; color: #90caf9; }}
  .ok {{ color: #69f0ae; }}
  .warn {{ color: #ff8a65; }}
  table {{ width: 100%; border-collapse: collapse;
           margin-top: 12px; }}
  th {{ background: #263045; padding: 8px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #263045; }}
  .badge-ok {{ background: #1b5e20; color: #69f0ae;
               padding: 2px 8px; border-radius: 4px; }}
  .badge-warn {{ background: #bf360c; color: #ff8a65;
                 padding: 2px 8px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>🧪 Stress & Bias Testing Report — Credit Risk Model</h1>
<p style="color:#90caf9">
  Generado: {time.strftime('%Y-%m-%d %H:%M:%S')} |
  API: {API_URL}
</p>

<h2>1. Stress Testing — Latencia</h2>
<div class="grid">
  <div class="card">
    <h3>Percentiles de Latencia por Test</h3>
    <canvas id="latencyChart"></canvas>
  </div>
  <div class="card">
    <h3>Tasa de Exito por Test</h3>
    <canvas id="successChart"></canvas>
  </div>
</div>

<h2>2. Bias Testing — Subgrupos</h2>
<div class="grid">
  <div class="card">
    <h3>Tasa de Rechazo por Sexo</h3>
    <canvas id="sexChart"></canvas>
  </div>
  <div class="card">
    <h3>Tasa de Rechazo por Tipo de Trabajo</h3>
    <canvas id="jobChart"></canvas>
  </div>
</div>

<h2>3. Contrafactuales</h2>
<div class="card">
  <h3>Delta de Probabilidad al Cambiar Variable Sensible</h3>
  <canvas id="cfChart" style="max-height:220px"></canvas>
</div>

<table style="margin-top:16px">
  <tr>
    <th>Test</th><th>Variable</th>
    <th>Prob A</th><th>Prob B</th>
    <th>Delta</th><th>Decision cambio</th><th>Sesgo</th>
  </tr>
  {''.join(
      "<tr>"
      f"<td>{r['test']}</td>"
      f"<td>{r['variable']}</td>"
      f"<td>{r['prob_a']}</td>"
      f"<td>{r['prob_b']}</td>"
      f"<td>{r['delta']:+.4f}</td>"
      f"<td>{'Si' if r['decision_cambio'] else 'No'}</td>"
      f"<td>"
      + ('<span class=badge-warn>SESGO</span>'
         if r['sesgo_detectado']
         else '<span class=badge-ok>OK</span>')
      + "</td></tr>"
      for r in counterfactual_results
    )}

</table>

<h2>4. Fairness — Disparate Impact Ratio</h2>
<div class="grid">
  <div class="card">
    <h3>DIR por Grupo (umbral 0.80)</h3>
    <canvas id="dirChart"></canvas>
  </div>
  <div class="card">
    <h3>Resumen Fairness</h3>
    <table>
      <tr><th>Grupo</th><th>DIR</th>
          <th>Gap</th><th>80% Rule</th></tr>
      {''.join(
        f"<tr><td>{g}</td>"
        f"<td>{m['disparate_impact_ratio']:.4f}</td>"
        f"<td>{m['paridad_demografica_gap']:.4f}</td>"
        + ('<span class=badge-ok>CUMPLE</span>'
           if m['cumple_80_rule']
           else '<span class=badge-warn>VIOLA</span>')
        + "</td></tr>"
        for g, m in fairness.items()
      )}
    </table>
  </div>
</div>

<script>
const latencyCtx = document.getElementById('latencyChart');
new Chart(latencyCtx, {{
  type: 'bar',
  data: {{
    labels: {stress_labels},
    datasets: [
      {{ label: 'P50', data: {stress_p50},
         backgroundColor: '#4fc3f7' }},
      {{ label: 'P95', data: {stress_p95},
         backgroundColor: '#ff8a65' }},
      {{ label: 'P99', data: {stress_p99},
         backgroundColor: '#ef5350' }},
    ]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#90caf9' }} }},
      y: {{ ticks: {{ color: '#90caf9' }},
            title: {{ display: true, text: 'ms',
                       color: '#90caf9' }} }}
    }}
  }}
}});

const successCtx = document.getElementById('successChart');
new Chart(successCtx, {{
  type: 'bar',
  data: {{
    labels: {stress_labels},
    datasets: [{{
      label: 'Tasa de exito (%)',
      data: {stress_success},
      backgroundColor: '#69f0ae'
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#90caf9' }} }},
      y: {{ min: 0, max: 100,
            ticks: {{ color: '#90caf9' }} }}
    }}
  }}
}});

const sexCtx = document.getElementById('sexChart');
new Chart(sexCtx, {{
  type: 'bar',
  data: {{
    labels: {sex_labels},
    datasets: [
      {{ label: 'Tasa rechazo (%)', data: {sex_rejection},
         backgroundColor: ['#4fc3f7','#ff80ab'] }},
    ]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#90caf9' }} }},
      y: {{ ticks: {{ color: '#90caf9' }} }}
    }}
  }}
}});

const jobCtx = document.getElementById('jobChart');
new Chart(jobCtx, {{
  type: 'bar',
  data: {{
    labels: {job_labels},
    datasets: [{{
      label: 'Tasa rechazo (%)',
      data: {job_rejection},
      backgroundColor: ['#ef5350','#ff8a65','#ffb74d','#69f0ae']
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#90caf9' }} }},
      y: {{ ticks: {{ color: '#90caf9' }} }}
    }}
  }}
}});

const cfCtx = document.getElementById('cfChart');
new Chart(cfCtx, {{
  type: 'bar',
  data: {{
    labels: {cf_labels},
    datasets: [{{
      label: 'Delta probabilidad (%)',
      data: {cf_delta},
      backgroundColor: {cf_delta}.map(v =>
        Math.abs(v) > 10 ? '#ef5350' : '#69f0ae'
      )
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#90caf9' }} }},
      y: {{ ticks: {{ color: '#90caf9' }},
            title: {{ display: true, text: 'Delta (%)',
                       color: '#90caf9' }} }}
    }}
  }}
}});

const dirCtx = document.getElementById('dirChart');
new Chart(dirCtx, {{
  type: 'bar',
  data: {{
    labels: {dir_labels},
    datasets: [{{
      label: 'DIR (min 0.80)',
      data: {dir_values},
      backgroundColor: {dir_values}.map(v =>
        v >= 0.80 ? '#69f0ae' : '#ef5350'
      )
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ labels: {{ color: '#e0e0e0' }} }},
      annotation: {{}}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#90caf9' }} }},
      y: {{ min: 0, max: 1.2,
            ticks: {{ color: '#90caf9' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"\n📊 Reporte HTML generado: {REPORT_PATH.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress & Bias Testing para Credit Risk API"
    )
    parser.add_argument(
        "--mode",
        choices=["stress", "bias", "all"],
        default="all",
    )
    parser.add_argument(
        "--pushgateway",
        default=PUSHGATEWAY,
        help="URL del Prometheus Pushgateway",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Workers para stress testing",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Requests para volume stress test",
    )
    args = parser.parse_args()

    stress_results: list[StressResult] = []
    bias_results: list[BiasResult] = []
    counterfactual_results: list[dict[str, Any]] = []
    fairness: dict[str, Any] = {}

    print(f"🚀 Iniciando tests contra {API_URL}")

    if args.mode in ["stress", "all"]:
        print("\n" + "="*50)
        print("STRESS TESTING")
        print("="*50)
        tester = StressTester()
        stress_results.append(
            tester.run_volume_test(
                n_requests=args.requests,
                workers=args.workers,
            )
        )
        stress_results.append(tester.run_corruption_test())
        stress_results.append(tester.run_drift_simulation())
        stress_results.append(
            tester.run_dos_simulation(
                n_requests=args.requests * 2,
                workers=args.workers * 2,
            )
        )

    if args.mode in ["bias", "all"]:
        print("\n" + "="*50)
        print("BIAS TESTING")
        print("="*50)
        btester = BiasTester()
        bias_results = btester.run_subgroup_analysis()
        counterfactual_results = btester.run_counterfactual_test()
        fairness = btester.run_fairness_metrics(bias_results)

    # Exportar a Pushgateway
    if stress_results or bias_results:
        push_metrics_to_prometheus(
            stress_results,
            bias_results,
            fairness,
            args.pushgateway,
        )

    # Generar reporte HTML
    generate_html_report(
        stress_results,
        bias_results,
        counterfactual_results,
        fairness,
    )

    print("\n✅ Testing completado.")


if __name__ == "__main__":
    main()
