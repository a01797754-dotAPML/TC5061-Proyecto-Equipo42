"""Exporta del Model Registry local el modelo de P1 que tiene el alias champion
a models/p1_champion/, para que el bootstrap de Compose lo registre en un clon limpio.

registro_origen.json guarda el linaje del run original (parámetros, métricas, commit,
data_version) y el historial de versiones y aliases, porque el Registry de Compose
empieza vacío y no conoce el run de origen."""

from datetime import datetime
import json
from pathlib import Path
import shutil

import mlflow
from mlflow import MlflowClient

from mibici_mlops import config

NOMBRE = config.MODELOS_REGISTRADOS["p1"]
ALIAS = "champion"
DESTINO = Path("models/p1_champion")
TAGS_LINAJE = (
    "equipo",
    "question",
    "git_commit",
    "git_cambios_sin_commit",
    "model_family",
    "stage_of_work",
)


def historial_versiones(client: MlflowClient) -> list[dict]:
    aliases = client.get_registered_model(NOMBRE).aliases
    historial = []
    for v in sorted(client.search_model_versions(f"name = '{NOMBRE}'"), key=lambda v: int(v.version)):
        historial.append(
            {
                "version": str(v.version),
                "run_id": v.run_id,
                "run_name": client.get_run(v.run_id).info.run_name if v.run_id else None,
                "aliases": sorted(a for a, ver in aliases.items() if ver == v.version),
                "creada": datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return historial


def metadatos_origen(client: MlflowClient, mv) -> dict:
    run = client.get_run(mv.run_id)
    return {
        "modelo": NOMBRE,
        "alias": ALIAS,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "run_name": run.info.run_name,
        "experimento": client.get_experiment(run.info.experiment_id).name,
        "source": mv.source,
        "descripcion": mv.description,
        "tracking_uri_origen": config.MLFLOW_TRACKING_URI,
        "tags_version": dict(mv.tags),
        "tags_run": {k: v for k, v in run.data.tags.items() if k in TAGS_LINAJE},
        "params": dict(run.data.params),
        "metricas": dict(run.data.metrics),
        "historial_versiones": historial_versiones(client),
    }


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(NOMBRE, ALIAS)

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        mlflow.artifacts.download_artifacts(
            f"models:/{NOMBRE}@{ALIAS}", dst_path=str(DESTINO.parent / "_tmp_export")
        )
    )
    shutil.move(str(tmp), DESTINO)
    shutil.rmtree(DESTINO.parent / "_tmp_export", ignore_errors=True)

    info = metadatos_origen(client, mv)
    (DESTINO / "registro_origen.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(f.stat().st_size for f in DESTINO.rglob("*") if f.is_file())
    print(f"Exportado {NOMBRE} v{mv.version} (@{ALIAS}, run {mv.run_id}) -> {DESTINO}")
    for f in sorted(DESTINO.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(DESTINO)}  {f.stat().st_size / 1024:.1f} KB")
    print(f"Tamaño total: {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
