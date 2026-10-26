"""Evaluación final reproducible del modelo ganador P2.

Uso: python -m mibici_mlops.modeling.p2_test [run_id]
Sin run_id evalúa la última búsqueda p2-rf-search terminada del experimento del equipo.
"""

import sys

import mlflow
from mlflow import MlflowClient

from mibici_mlops.entrenar_p2 import evaluar_predicciones
from mibici_mlops.modeling.p2_cache import cargar_o_preparar_p2
from mibici_mlops.tracking import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def ultimo_run_busqueda_p2() -> str:
    """run_id de la última búsqueda p2-rf-search terminada (el ganador por validación)."""
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string="tags.mlflow.runName = 'p2-rf-search' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise ValueError("No hay una búsqueda p2-rf-search terminada: correr model=rf-search antes")
    return runs[0].info.run_id


def evaluar_test_p2(run_id: str) -> dict:
    """Evalúa en test un modelo P2 ya seleccionado por validación."""
    client = MlflowClient()
    run = client.get_run(run_id)

    model_uri = run.data.tags.get("model_uri")
    if not model_uri:
        raise ValueError(
            f"El run {run_id} no contiene el tag model_uri"
        )

    print(f"Cargando modelo: {model_uri}", flush=True)
    modelo = mlflow.sklearn.load_model(model_uri)

    _, _, _, _, X_test, y_test = cargar_o_preparar_p2()

    print("Evaluando modelo ganador sobre test...", flush=True)
    predicciones = modelo.predict(X_test)

    metricas = evaluar_predicciones(
        y_test,
        predicciones,
        conjunto="test",
    )

    for nombre, valor in metricas.items():
        client.log_metric(run_id, nombre, float(valor))

    print(f"Métricas de test: {metricas}", flush=True)
    return metricas


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    evaluar_test_p2(sys.argv[1] if len(sys.argv) > 1 else ultimo_run_busqueda_p2())
