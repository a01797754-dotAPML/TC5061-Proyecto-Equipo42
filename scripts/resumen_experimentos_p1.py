"""Resume los runs de P1 del experimento del equipo (sin los hijos de las búsquedas)
en docs/evidencias/actividad11_experimentos_p1.csv, como evidencia de cómo se eligió el champion.

    python scripts/resumen_experimentos_p1.py
"""

from datetime import datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import pandas as pd

from mibici_mlops import config

SALIDA = Path("docs/evidencias/actividad11_experimentos_p1.csv")


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], max_results=5000)

    filas = []
    for r in runs:
        tags = r.data.tags
        pregunta = tags.get("question") or tags.get("pregunta") or ""
        nombre = r.info.run_name or ""
        if pregunta.upper() != "P1" and not nombre.startswith("p1-"):
            continue
        if "mlflow.parentRunId" in tags:
            continue
        hijos = client.search_runs(
            [exp.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{r.info.run_id}'"
        )
        filas.append(
            {
                "run_name": nombre,
                "inicio": datetime.fromtimestamp(r.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                "stage_of_work": tags.get("stage_of_work", ""),
                "model_family": tags.get("model_family", ""),
                "runs_hijos": len(hijos),
                "data_version": r.data.params.get("data_version", ""),
                "git_commit": tags.get("git_commit", "")[:7],
                "run_id": r.info.run_id,
                **{k: round(v, 4) for k, v in sorted(r.data.metrics.items())},
            }
        )

    df = pd.DataFrame(filas).sort_values("inicio")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.drop(columns=["run_id"]).to_string(index=False))
    print(f"\n{len(df)} runs de P1 -> {SALIDA}")


if __name__ == "__main__":
    main()
