import asyncio
import time
from typing import Optional, List
import httpx
from tests.test_load import generate_random_data

URL: str = "http://localhost:8000/predict"
TOTAL_REQUESTS: int = 50


async def send_request(client: httpx.AsyncClient) -> Optional[float]:
    """Envía una petición asíncrona y devuelve la latencia."""
    payload: dict = {"data": generate_random_data()}
    try:
        start: float = time.perf_counter()
        response: httpx.Response = await client.post(URL, json=payload)
        latency: float = time.perf_counter() - start
        if response.status_code == 200:
            return latency
        return None
    except Exception:
        return None


async def run_stress_test() -> None:
    """Ejecuta el bloque de peticiones concurrentes."""
    print(f"🚀 Iniciando Stress Test: {TOTAL_REQUESTS} peticiones a {URL}...")

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Usamos list comprehension con anotación explícita
        tasks = [send_request(client) for _ in range(TOTAL_REQUESTS)]
        results: List[Optional[float]] = await asyncio.gather(*tasks)

    latencies: List[float] = [r for r in results if r is not None]

    if latencies:
        avg_lat: float = sum(latencies) / len(latencies)
        print("\n✅ Test completado con éxito.")
        print(f"📊 Peticiones exitosas: {len(latencies)}/{TOTAL_REQUESTS}")
        print(f"⏱️ Latencia promedio: {avg_lat:.4f} seg")
        print(f"⚡ Petición más rápida: {min(latencies):.4f} seg")
        print(f"🐢 Petición más lenta: {max(latencies):.4f} seg")
    else:
        print("\n❌ Todas las peticiones fallaron. Revisa los logs de Docker.")


if __name__ == "__main__":
    asyncio.run(run_stress_test())
