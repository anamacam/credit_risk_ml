import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List
from src.credit_risk_analysis.config.config_loader import config

import aiofiles  # type: ignore
import httpx

# Configuración de la prueba
API_URL: str = config.get("test_api_url", "http://localhost:8000/predict")
TOTAL_REQUESTS: int = 500
CONCURRENT_REQUESTS: int = 50


def generate_random_data() -> Dict[str, Any]:
    """Genera datos dinámicos basados en las reglas del schema.yaml."""
    payload: Dict[str, Any] = {}
    features = config.get("features", [])

    for feat in features:
        name = feat["name"]
        f_type = feat["type"]

        if f_type in ("slider", "number"):
            payload[name] = random.randint(feat["min"], feat["max"])
        elif f_type == "selectbox":
            payload[name] = random.choice(feat["options"])

    return payload


async def send_request(client: httpx.AsyncClient) -> float:
    """Envía una petición POST y devuelve la latencia."""
    payload: Dict[str, Any] = generate_random_data()
    start_time: float = time.perf_counter()

    try:
        response: httpx.Response = await client.post(
            API_URL,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return -1.0

    latency: float = time.perf_counter() - start_time
    return latency


async def run_load_test() -> None:
    """Ejecuta el test de carga y guarda resultados de forma asíncrona."""
    print("\n🚀 Stress test iniciado (Dinámico via YAML)")
    print(f"Target: {API_URL}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Concurrencia: {CONCURRENT_REQUESTS}\n")

    latencies: List[float] = []

    async with httpx.AsyncClient() as client:
        semaphore: asyncio.Semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def limited_request() -> None:
            async with semaphore:
                latency: float = await send_request(client)
                latencies.append(latency)

        tasks: List[asyncio.Task[None]] = [
            asyncio.create_task(limited_request())
            for _ in range(TOTAL_REQUESTS)
        ]

        start_time: float = time.perf_counter()
        await asyncio.gather(*tasks)
        total_time: float = time.perf_counter() - start_time

    valid_latencies = [lat for lat in latencies if lat > 0]
    errors = len(latencies) - len(valid_latencies)

    avg_latency = (
        sum(valid_latencies) / len(valid_latencies)
        if valid_latencies else 0.0
    )
    rps = TOTAL_REQUESTS / total_time

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_requests": TOTAL_REQUESTS,
        "concurrency": CONCURRENT_REQUESTS,
        "total_time_seconds": round(total_time, 2),
        "avg_latency_seconds": round(avg_latency, 3),
        "requests_per_second": round(rps, 2),
        "errors": errors
    }

    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    async with aiofiles.open(filename, mode="w") as f:
        await f.write(json.dumps(report, indent=4))

    print("\n📊 RESULTADOS")
    print("----------------------")
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"Requests por segundo: {rps:.2f}")
    print(f"Latencia promedio: {avg_latency:.3f}s")
    print(f"Errores: {errors}")
    print(f"Reporte guardado en: {filename}")


def main() -> None:
    """Punto de entrada."""
    asyncio.run(run_load_test())


if __name__ == "__main__":
    main()
