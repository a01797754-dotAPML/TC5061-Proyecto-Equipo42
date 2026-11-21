"""Prueba de carga de la API de P1 (demanda MiBici por estación y hora).

Escenario: estaciones consultando la predicción de la siguiente hora para el
rebalanceo en tiempo real. Cada usuario virtual envía filas reales de sep-2026
(load_tests/payloads.json) a POST /predict y, de vez en cuando, revisa GET /health.

Uso (headless, reproducible):
    locust -f load_tests/locustfile.py --host http://localhost:8000 \
        --headless -u 200 -r 10 -t 6m --csv reports/locust/run1 --html reports/locust/run1.html
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

PAYLOADS = json.loads((Path(__file__).parent / "payloads.json").read_text())


class EstacionUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        with self.client.post(
            "/predict", json=random.choice(PAYLOADS), name="POST /predict", catch_response=True
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif "prediction" not in r.json():
                r.failure("respuesta sin 'prediction'")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
