"""Exporta del Model Registry local el modelo de P1 que tiene el alias champion
a models/p1_champion/, para que el bootstrap de Compose lo registre en un clon limpio."""

import json
from pathlib import Path
import shutil

import mlflow
from mlflow import MlflowClient

from mibici_mlops import config

NOMBRE = config.MODELOS_REGISTRADOS["p1"]
ALIAS = "champion"
DESTINO = Path("models/p1_champion")


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

    info = {
        "modelo": NOMBRE,
        "alias": ALIAS,
        "version": mv.version,
        "run_id": mv.run_id,
        "source": mv.source,
        "descripcion": mv.description,
        "tracking_uri_origen": config.MLFLOW_TRACKING_URI,
    }
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
