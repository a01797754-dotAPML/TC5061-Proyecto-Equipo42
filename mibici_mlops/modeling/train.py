"""Punto de entrada `train` del MLflow Project: entrena un candidato de P1, P2 o P3.

Recibe la pregunta, el modelo y sus hiperparámetros (JSON) y llama a
`mibici_mlops.modeling.<pregunta>.train(model, params)`, que registra la corrida en MLflow.

Uso (desde la raíz del repo, con el servidor de tracking levantado):
    python -m mlflow run . -e train --experiment-name mibici-equipo42 \
        -P question=p1 -P model=<modelo> -P params='{"<hiperparametro>": <valor>}'
También funciona sin MLflow Project, para pruebas locales:
    python -m mibici_mlops.modeling.train --question p1 --model <modelo> --params '{...}'

Contrato: cada mibici_mlops/modeling/pN.py expone `train(model: str, params: dict) -> str` (run_id)
y registra la corrida con tracking.iniciar_run + tracking.registrar_resultados.
"""

import importlib
import json

from loguru import logger
import typer

PREGUNTAS = ("p1", "p2", "p3")

app = typer.Typer()


def leer_params(texto: str) -> dict:
    """Convierte el JSON de hiperparámetros en diccionario y da un error claro si está mal escrito."""
    try:
        params = json.loads(texto or "{}")
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"--params no es JSON válido: {e}") from e
    if not isinstance(params, dict):
        raise typer.BadParameter(
            "--params debe ser un objeto JSON, por ejemplo '{\"max_iter\": 300}'"
        )
    return params


@app.command()
def main(
    question: str = typer.Option("p1", help="Pregunta: p1, p2 o p3."),
    model: str = typer.Option(..., help="Modelo, según los que implemente la pregunta."),
    params: str = typer.Option(
        "{}", help="Hiperparámetros en JSON, por ejemplo '{\"max_iter\": 300}'."
    ),
):
    question = question.strip().lower()
    if question not in PREGUNTAS:
        raise typer.BadParameter(f"Pregunta desconocida: {question}. Opciones: {PREGUNTAS}")

    modulo = importlib.import_module(f"mibici_mlops.modeling.{question}")
    if not hasattr(modulo, "train"):
        raise typer.BadParameter(f"{question} todavía no implementa train(model, params)")

    hiperparametros = leer_params(params)
    logger.info(f"Entrenando {question} · modelo={model} · params={hiperparametros}")
    run_id = modulo.train(model, hiperparametros)
    logger.success(f"{question} terminado: run {run_id}")


if __name__ == "__main__":
    app()
