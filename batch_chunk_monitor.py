"""
Batch Chunk Monitor — Credit Risk API

Envia 100 perfiles en chunks de 10 al endpoint /predict-batch,
muestra progreso dinamico en consola y empuja metricas a
Prometheus Pushgateway para visualizacion en tiempo real en Grafana.

Uso:
    python batch_chunk_monitor.py
    python batch_chunk_monitor.py --chunks 10 --delay 1.5
    python batch_chunk_monitor.py --pushgateway http://localhost:9091
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
API_URL: str = os.getenv("API_URL", "http://localhost:8000")
PUSHGATEWAY: str = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
BATCH_URL: str = f"{API_URL}/predict-batch"
CHUNK_SIZE: int = 10
TOTAL_PROFILES: int = 100

# Colores ANSI para consola
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\033[1G"

# ---------------------------------------------------------------------------
# Generador de perfiles de prueba (100 perfiles variados)
# ---------------------------------------------------------------------------
def generate_profiles(n: int = 100) -> list[dict[str, Any]]:
    """Genera n perfiles de credito con variedad de riesgo."""
    import random
    rng = random.Random(42)

    sexos = ["male", "female"]
    viviendas = ["own", "rent", "free"]
    ahorros = ["little", "moderate", "rich", "quite rich"]
    cuentas = ["little", "moderate", "rich"]
    propositos = [
        "car", "business", "furniture/equipment",
        "radio/TV", "education", "repairs",
        "domestic appliances", "vacation/others",
    ]

    profiles = []
    for i in range(n):
        # Alternamos entre perfiles de bajo y alto riesgo
        if i % 3 == 0:
            # Alto riesgo
            profile: dict[str, Any] = {
                "age": rng.randint(18, 30),
                "sex": rng.choice(sexos),
                "job": rng.choice([0, 1]),
                "housing": "rent",
                "saving_accounts": "little",
                "checking_account": "little",
                "credit_amount": rng.randint(8000, 20000),
                "duration": rng.randint(48, 72),
                "purpose": rng.choice(["business", "vacation/others"]),
                "threshold": 0.30,
            }
        elif i % 3 == 1:
            # Bajo riesgo
            profile = {
                "age": rng.randint(40, 65),
                "sex": rng.choice(sexos),
                "job": rng.choice([2, 3]),
                "housing": "own",
                "saving_accounts": rng.choice(["moderate", "rich"]),
                "checking_account": rng.choice(["moderate", "rich"]),
                "credit_amount": rng.randint(1000, 6000),
                "duration": rng.randint(6, 24),
                "purpose": rng.choice(["car", "repairs"]),
                "threshold": 0.30,
            }
        else:
            # Riesgo medio
            profile = {
                "age": rng.randint(25, 50),
                "sex": rng.choice(sexos),
                "job": rng.randint(0, 3),
                "housing": rng.choice(viviendas),
                "saving_accounts": rng.choice(ahorros),
                "checking_account": rng.choice(cuentas),
                "credit_amount": rng.randint(2000, 12000),
                "duration": rng.randint(12, 48),
                "purpose": rng.choice(propositos),
                "threshold": 0.30,
            }
        profiles.append(profile)
    return profiles


# ---------------------------------------------------------------------------
# Prometheus Pushgateway
# ---------------------------------------------------------------------------
def push_chunk_metrics(
    chunk_num: int,
    chunk_results: list[dict[str, Any]],
    cumulative: dict[str, Any],
    pushgateway_url: str,
) -> None:
    """Empuja metricas del chunk actual al Pushgateway."""
    high = sum(
        1 for r in chunk_results if r.get("decision") == "high_risk"
    )
    low = len(chunk_results) - high
    probs = [
        r["probability"] for r in chunk_results
        if "probability" in r
    ]
    avg_prob = sum(probs) / len(probs) if probs else 0.0
    max_prob = max(probs) if probs else 0.0
    min_prob = min(probs) if probs else 0.0

    lines = [
        f'batch_chunk_number {chunk_num}',
        f'batch_chunk_high_risk {high}',
        f'batch_chunk_low_risk {low}',
        f'batch_chunk_avg_probability {avg_prob:.4f}',
        f'batch_chunk_max_probability {max_prob:.4f}',
        f'batch_chunk_min_probability {min_prob:.4f}',
        f'batch_cumulative_total {cumulative["total"]}',
        f'batch_cumulative_high {cumulative["high"]}',
        f'batch_cumulative_low {cumulative["low"]}',
        f'batch_cumulative_avg_prob {cumulative["avg_prob"]:.4f}',
        f'batch_progress_pct {cumulative["pct"]:.1f}',
        f'batch_chunk_latency_ms {cumulative["last_latency"]:.1f}',
    ]

    payload = "\n".join(lines) + "\n"
    try:
        requests.post(
            f"{pushgateway_url}/metrics/job/batch_monitor"
            f"/chunk/{chunk_num}",
            data=payload,
            headers={"Content-Type": "text/plain"},
            timeout=3,
        )
    except Exception:
        pass  # No bloquear si pushgateway no disponible


# ---------------------------------------------------------------------------
# Visualizacion en consola
# ---------------------------------------------------------------------------
def print_header() -> None:
    """Imprime el header del monitor."""
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(
        f"{BOLD}{CYAN}  Credit Risk — Batch Chunk Monitor{RESET}"
    )
    print(
        f"{CYAN}  API: {WHITE}{API_URL}{RESET}"
    )
    print(
        f"{CYAN}  Pushgateway: {WHITE}{PUSHGATEWAY}{RESET}"
    )
    print(f"{BOLD}{CYAN}{'='*65}{RESET}\n")


def print_progress_bar(
    current: int,
    total: int,
    width: int = 40,
) -> str:
    """Genera barra de progreso ASCII."""
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if pct >= 0.8 else YELLOW if pct >= 0.4 else CYAN
    return f"{color}[{bar}]{RESET} {pct:.0%}"


def print_chunk_result(
    chunk_num: int,
    total_chunks: int,
    profiles_sent: int,
    results: list[dict[str, Any]],
    latency_ms: float,
    cumulative: dict[str, Any],
) -> None:
    """Imprime resultado detallado de un chunk."""
    high = sum(
        1 for r in results if r.get("decision") == "high_risk"
    )
    low = len(results) - high
    probs = [r["probability"] for r in results if "probability" in r]
    avg_prob = sum(probs) / len(probs) if probs else 0.0

    # Barra de progreso
    progress = print_progress_bar(profiles_sent, TOTAL_PROFILES)

    # Header del chunk
    print(
        f"\n{BOLD}{BLUE}► Chunk {chunk_num}/{total_chunks}{RESET}"
        f"  {progress}"
    )
    print(
        f"  {WHITE}Perfiles:{RESET} {profiles_sent}/{TOTAL_PROFILES}"
        f"  {WHITE}Latencia:{RESET} {YELLOW}{latency_ms:.0f}ms{RESET}"
    )

    # Mini tabla de resultados del chunk
    print(
        f"  {WHITE}{'Perfil':<8} {'Prob':>8} {'Decision':<12} "
        f"{'Barra':<20}{RESET}"
    )
    print(f"  {WHITE}{'─'*52}{RESET}")

    for i, r in enumerate(results):
        prob = r.get("probability", 0.0)
        decision = r.get("decision", "error")
        bar_len = int(prob * 20)
        bar = "▓" * bar_len + "░" * (20 - bar_len)

        if decision == "high_risk":
            color = RED
            symbol = "✗"
        elif decision == "low_risk":
            color = GREEN
            symbol = "✓"
        else:
            color = YELLOW
            symbol = "?"

        print(
            f"  {WHITE}#{(chunk_num-1)*10+i+1:<6}{RESET}"
            f" {color}{prob:>8.4f}{RESET}"
            f" {color}{symbol} {decision:<10}{RESET}"
            f" {color}{bar}{RESET}"
        )

    # Resumen del chunk
    print(f"  {WHITE}{'─'*52}{RESET}")
    risk_pct = (high / len(results) * 100) if results else 0
    print(
        f"  {WHITE}Chunk:{RESET}"
        f"  {RED}Alto riesgo: {high}{RESET}"
        f"  {GREEN}Bajo riesgo: {low}{RESET}"
        f"  {WHITE}Prob media: {YELLOW}{avg_prob:.4f}{RESET}"
    )

    # Resumen acumulado
    cum_pct = (
        cumulative["high"] / cumulative["total"] * 100
        if cumulative["total"] > 0 else 0
    )
    print(
        f"\n  {BOLD}{WHITE}Acumulado:{RESET}"
        f"  Total: {CYAN}{cumulative['total']}{RESET}"
        f"  {RED}Rechazados: {cumulative['high']}{RESET}"
        f"  {GREEN}Aprobados: {cumulative['low']}{RESET}"
        f"  {WHITE}Tasa rechazo: "
        f"{YELLOW}{cum_pct:.1f}%{RESET}"
    )

    # Alerta de drift
    if risk_pct > 70:
        print(
            f"\n  {BOLD}{RED}⚠ ALERTA: "
            f"{risk_pct:.0f}% alto riesgo en este chunk{RESET}"
        )


def print_final_summary(
    all_results: list[dict[str, Any]],
    total_time: float,
    total_chunks: int,
    chunk_latencies: list[float],
) -> None:
    """Imprime resumen final con estadisticas completas."""
    high = sum(
        1 for r in all_results if r.get("decision") == "high_risk"
    )
    low = len(all_results) - high
    probs = [r["probability"] for r in all_results if "probability" in r]
    avg_prob = sum(probs) / len(probs) if probs else 0.0
    max_prob = max(probs) if probs else 0.0
    min_prob = min(probs) if probs else 0.0
    avg_latency = (
        sum(chunk_latencies) / len(chunk_latencies)
        if chunk_latencies else 0
    )

    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  RESUMEN FINAL{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")
    print(
        f"\n  {WHITE}Total perfiles procesados:{RESET}"
        f"  {CYAN}{len(all_results)}{RESET}"
    )
    print(
        f"  {WHITE}Tiempo total:{RESET}"
        f"  {YELLOW}{total_time:.2f}s{RESET}"
    )
    print(
        f"  {WHITE}Chunks procesados:{RESET}"
        f"  {CYAN}{total_chunks}{RESET}"
    )
    print(
        f"  {WHITE}Latencia media por chunk:{RESET}"
        f"  {YELLOW}{avg_latency:.0f}ms{RESET}"
    )
    print(f"\n  {BOLD}{WHITE}Clasificacion:{RESET}")
    print(
        f"  {RED}  Alto riesgo (rechazados):"
        f"  {high} ({high/len(all_results)*100:.1f}%){RESET}"
    )
    print(
        f"  {GREEN}  Bajo riesgo (aprobados): "
        f"  {low} ({low/len(all_results)*100:.1f}%){RESET}"
    )
    print(f"\n  {BOLD}{WHITE}Probabilidades:{RESET}")
    print(
        f"  {WHITE}  Media:  {YELLOW}{avg_prob:.4f}{RESET}"
    )
    print(
        f"  {WHITE}  Maxima: {RED}{max_prob:.4f}{RESET}"
    )
    print(
        f"  {WHITE}  Minima: {GREEN}{min_prob:.4f}{RESET}"
    )

    # Grafico de distribucion ASCII
    print(f"\n  {BOLD}{WHITE}Distribucion de scores:{RESET}")
    buckets = [0] * 10
    for p in probs:
        idx = min(int(p * 10), 9)
        buckets[idx] += 1
    max_b = max(buckets) if buckets else 1
    for i, count in enumerate(buckets):
        bar = "▓" * int(count / max_b * 30)
        label = f"{i*0.1:.1f}-{(i+1)*0.1:.1f}"
        color = RED if i >= 3 else GREEN
        print(
            f"  {WHITE}{label}{RESET}"
            f" {color}{bar:<30}{RESET}"
            f" {WHITE}{count}{RESET}"
        )

    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(
        f"{BOLD}{GREEN}  ✓ Batch completado exitosamente{RESET}"
    )
    print(
        f"{CYAN}  Dashboard Grafana: "
        f"{WHITE}http://localhost:3000{RESET}"
    )
    print(f"{BOLD}{CYAN}{'='*65}{RESET}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Chunk Monitor para Credit Risk API"
    )
    parser.add_argument(
        "--chunks", type=int, default=10,
        help="Numero de chunks (default: 10)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Segundos entre chunks (default: 1.0)"
    )
    parser.add_argument(
        "--pushgateway", default=PUSHGATEWAY,
        help="URL del Prometheus Pushgateway"
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Desactivar colores en consola"
    )
    args = parser.parse_args()

    print_header()

    # Generar 100 perfiles
    profiles = generate_profiles(TOTAL_PROFILES)
    chunk_size = TOTAL_PROFILES // args.chunks
    chunks = [
        profiles[i:i + chunk_size]
        for i in range(0, TOTAL_PROFILES, chunk_size)
    ]

    all_results: list[dict[str, Any]] = []
    chunk_latencies: list[float] = []
    cumulative: dict[str, Any] = {
        "total": 0,
        "high": 0,
        "low": 0,
        "avg_prob": 0.0,
        "pct": 0.0,
        "last_latency": 0.0,
    }

    print(
        f"{CYAN}Enviando {TOTAL_PROFILES} perfiles en "
        f"{len(chunks)} chunks de {chunk_size}...{RESET}\n"
    )

    start_total = time.perf_counter()

    for i, chunk in enumerate(chunks, 1):
        start = time.perf_counter()

        try:
            resp = requests.post(
                BATCH_URL,
                json={"profiles": chunk},
                timeout=30,
            )
            latency = (time.perf_counter() - start) * 1000
            chunk_latencies.append(latency)

            if resp.status_code == 200:
                data = resp.json()
                results: list[dict[str, Any]] = data.get(
                    "results", []
                )
                all_results.extend(results)

                # Actualizar acumulado
                cumulative["total"] = len(all_results)
                cumulative["high"] = sum(
                    1 for r in all_results
                    if r.get("decision") == "high_risk"
                )
                cumulative["low"] = (
                    cumulative["total"] - cumulative["high"]
                )
                probs_cum = [
                    r["probability"] for r in all_results
                    if "probability" in r
                ]
                cumulative["avg_prob"] = (
                    sum(probs_cum) / len(probs_cum)
                    if probs_cum else 0.0
                )
                cumulative["pct"] = (
                    cumulative["total"] / TOTAL_PROFILES * 100
                )
                cumulative["last_latency"] = latency

                print_chunk_result(
                    chunk_num=i,
                    total_chunks=len(chunks),
                    profiles_sent=cumulative["total"],
                    results=results,
                    latency_ms=latency,
                    cumulative=cumulative,
                )

                # Push a Prometheus
                push_chunk_metrics(
                    chunk_num=i,
                    chunk_results=results,
                    cumulative=cumulative,
                    pushgateway_url=args.pushgateway,
                )

            else:
                print(
                    f"\n{RED}Error en chunk {i}: "
                    f"HTTP {resp.status_code}{RESET}"
                )

        except Exception as exc:
            print(f"\n{RED}Error en chunk {i}: {exc}{RESET}")

        # Delay entre chunks para ver el dashboard moverse
        if i < len(chunks):
            print(
                f"\n  {YELLOW}Esperando {args.delay}s "
                f"antes del siguiente chunk...{RESET}"
            )
            time.sleep(args.delay)

    total_time = time.perf_counter() - start_total
    print_final_summary(
        all_results=all_results,
        total_time=total_time,
        total_chunks=len(chunks),
        chunk_latencies=chunk_latencies,
    )


if __name__ == "__main__":
    main()
