"""Registra en el MLflow de Compose el champion de P1 exportado en models/p1_champion/.
Si el alias champion ya existe (volumen persistente), no hace nada.

El linaje del run original (parámetros, métricas, commit y data_version) se copia de
registro_origen.json al run de bootstrap y a la versión registrada, porque el contenedor
no tiene Git ni acceso al MLflow donde se entrenó el modelo."""

import json
import os
from pathlib import Path
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

NOMBRE = os.getenv("MODEL_NAME", "mibici-equipo42-p1-demanda")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
CARPETA = Path(os.getenv("MODEL_DIR", "models/p1_champion"))
DESCRIPCION_MODELO = (
    "P1 - Viajes que inician por estación en la siguiente hora (regresión). "
    "HistGradientBoosting con pérdida Poisson."
)


def tags_linaje(origen: dict) -> dict:
    tags_run = origen.get("tags_run", {})
    params = origen.get("params", {})
    return {
        "origen_version": str(origen["version"]),
        "origen_run_id": origen["run_id"],
        "origen_run_name": origen.get("run_name", ""),
        "origen_experimento": origen.get("experimento", ""),
        "git_commit": tags_run.get("git_commit", ""),
        "data_version": params.get("data_version", ""),
        "question": tags_run.get("question", "P1"),
    }


def descripcion_version(origen: dict) -> str:
    metricas = origen.get("metricas", {})
    params = origen.get("params", {})
    texto = (
        f"Bootstrap de Compose: copia de la versión {origen['version']} "
        f"(run {origen['run_id']}) del Registry del equipo."
    )
    if params.get("entrenado_con"):
        texto += f" Entrenado con {params['entrenado_con']} (data_version {params.get('data_version', '?')})."
    if "test_sep_mae" in metricas:
        texto += (
            f" A/B en sep-2026: MAE {metricas['test_sep_mae']:.4f},"
            f" RMSE {metricas['test_sep_rmse']:.4f}."
        )
    return texto


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(NOMBRE, ALIAS)
        print(f"{NOMBRE}@{ALIAS} ya existe (versión {mv.version}); no se registra de nuevo")
        return
    except MlflowException:
        pass

    if not (CARPETA / "MLmodel").exists():
        sys.exit(f"No se encontró el modelo exportado en {CARPETA}")
    origen = json.loads((CARPETA / "registro_origen.json").read_text(encoding="utf-8"))
    linaje = tags_linaje(origen)

    mlflow.set_experiment("mibici-equipo42")
    with mlflow.start_run(run_name="p1-champion-bootstrap") as run:
        mlflow.set_tags({"stage_of_work": "bootstrap", **linaje})
        mlflow.log_params(origen.get("params", {}))
        mlflow.log_metrics(origen.get("metricas", {}))
        mlflow.log_dict(origen, "linaje/registro_origen.json")
        mlflow.log_artifacts(str(CARPETA), artifact_path="model")

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", NOMBRE, tags=linaje)
    client.update_registered_model(NOMBRE, description=DESCRIPCION_MODELO)
    client.set_registered_model_alias(NOMBRE, ALIAS, mv.version)
    client.update_model_version(NOMBRE, mv.version, description=descripcion_version(origen))
    print(f"Registrado {NOMBRE} versión {mv.version} con alias {ALIAS}")
    print(f"Linaje: {json.dumps(linaje, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
