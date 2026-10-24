"""Evaluación final reproducible del modelo ganador P2."""

import mlflow
from mlflow import MlflowClient

from mibici_mlops.entrenar_p2 import evaluar_predicciones
from mibici_mlops.modeling.p2_cache import cargar_o_preparar_p2


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
    evaluar_test_p2("754834f5c4294e7eac70e73b6406772d")
