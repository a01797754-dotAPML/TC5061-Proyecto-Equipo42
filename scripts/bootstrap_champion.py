"""Registra en el MLflow de Compose el champion de P1 exportado en models/p1_champion/.
Si el alias champion ya existe (volumen persistente), no hace nada."""

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

    mlflow.set_experiment("mibici-equipo42")
    with mlflow.start_run(run_name="p1-champion-bootstrap") as run:
        mlflow.set_tags(
            {
                "stage_of_work": "bootstrap",
                "origen_version": origen["version"],
                "origen_run_id": origen["run_id"],
            }
        )
        mlflow.log_artifacts(str(CARPETA), artifact_path="model")

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", NOMBRE)
    client.set_registered_model_alias(NOMBRE, ALIAS, mv.version)
    client.update_model_version(
        NOMBRE,
        mv.version,
        description=f"Bootstrap de Compose: copia de la versión {origen['version']} "
        f"(run {origen['run_id']}) del Registry del equipo.",
    )
    print(f"Registrado {NOMBRE} versión {mv.version} con alias {ALIAS}")


if __name__ == "__main__":
    main()
