"""Pruebas de las piezas compartidas de la entrega 2: registro común y punto de entrada `train`.

No usan datos de MiBici ni modelos de ninguna pregunta: un Ridge sobre datos sintéticos basta
para comprobar el contrato que usan P1, P2 y P3. MLflow escribe en una base SQLite temporal.
"""

import types

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from typer.testing import CliRunner

from mibici_mlops import tracking
from mibici_mlops.modeling import train as entrada_train


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """Apunta el registro común a una base SQLite temporal y cierra cualquier run abierto."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def datos():
    X = pd.DataFrame({"x1": np.arange(40, dtype=float), "x2": np.arange(40, dtype=float) % 5})
    return X, 2 * X["x1"] + X["x2"]


def test_run_candidato_registra_tags_params_y_artefactos(mlflow_temporal, datos, tmp_path):
    X, y = datos
    grafica = tmp_path / "ejemplo.png"
    grafica.write_bytes(b"png de prueba")
    modelo = Ridge(alpha=1.0).fit(X, y)

    with tracking.iniciar_run(
        "P1", "p1-ejemplo", "ridge", "candidate", params={"alpha": 1.0}
    ) as run:
        model_uri = tracking.registrar_resultados(
            {"val_mae": 0.1, "test_mae": 0.2},
            modelo=modelo,
            X_ejemplo=X,
            graficas=[grafica],
            config={"modelo": "ridge"},
            datos_entrenamiento=X.assign(y=y),
            nombre_datos="ejemplo-train",
        )

    r = mlflow.get_run(run.info.run_id)
    assert r.info.run_name == "p1-ejemplo"
    for tag, valor in {
        "question": "P1",
        "model_family": "ridge",
        "stage_of_work": "candidate",
    }.items():
        assert r.data.tags[tag] == valor
    assert {"git_commit", "equipo", "model_uri"} <= set(r.data.tags)
    assert {"data_version", "seed", "val_inicio", "test_inicio", "alpha"} <= set(r.data.params)

    artefactos = {a.path for a in mlflow.artifacts.list_artifacts(run_id=r.info.run_id)}
    assert {"plots", "config", "data"} <= artefactos
    assert [d.dataset.name for d in r.inputs.dataset_inputs] == ["ejemplo-train"]

    cargado = mlflow.pyfunc.load_model(model_uri)
    assert cargado.metadata.signature is not None
    assert len(cargado.predict(X.head(3))) == 3


def test_busqueda_con_nested_runs(mlflow_temporal, datos):
    X, y = datos
    with tracking.iniciar_run("P1", "p1-ejemplo-search", "ridge", "search") as padre:
        for alpha in (0.1, 1.0):
            with tracking.iniciar_run(
                "P1",
                f"p1-ejemplo-a{alpha}",
                "ridge",
                "search",
                params={"alpha": alpha},
                nested=True,
            ):
                modelo = Ridge(alpha=alpha).fit(X, y)
                tracking.registrar_resultados(
                    {"val_mae": float(alpha)}, registrar_manifiesto=False
                )

    hijos = mlflow.search_runs(
        experiment_ids=[padre.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{padre.info.run_id}'",
        output_format="list",
    )
    assert sorted(h.info.run_name for h in hijos) == ["p1-ejemplo-a0.1", "p1-ejemplo-a1.0"]
    assert modelo is not None


def test_valida_stage_metricas_y_datos_personales(mlflow_temporal):
    with (
        pytest.raises(ValueError, match="stage_of_work"),
        tracking.iniciar_run("P1", "x", "ridge", "candidato"),
    ):
        pass
    with (
        pytest.raises(ValueError, match="datos personales"),
        tracking.iniciar_run("P1", "x", "ridge", "candidate", params={"usuario_id": 1}),
    ):
        pass
    with (
        tracking.iniciar_run("P1", "x", "ridge", "candidate"),
        pytest.raises(ValueError, match="prefijo"),
    ):
        tracking.registrar_resultados({"mae": 0.1}, registrar_manifiesto=False)


def test_entrada_train_despacha_a_la_pregunta(monkeypatch):
    llamadas = {}

    def train_falso(model, params):
        llamadas.update(model=model, params=params)
        return "run-falso"

    monkeypatch.setattr(
        entrada_train.importlib,
        "import_module",
        lambda nombre: types.SimpleNamespace(train=train_falso),
    )
    resultado = CliRunner().invoke(
        entrada_train.app,
        ["--question", "P2", "--model", "rf", "--params", '{"n_estimators": 10}'],
    )
    assert resultado.exit_code == 0, resultado.output
    assert llamadas == {"model": "rf", "params": {"n_estimators": 10}}


def test_entrada_train_rechaza_errores_de_uso(monkeypatch):
    runner = CliRunner()
    assert runner.invoke(entrada_train.app, ["--question", "p4", "--model", "rf"]).exit_code != 0
    assert (
        runner.invoke(
            entrada_train.app, ["--question", "p1", "--model", "rf", "--params", "{mal"]
        ).exit_code
        != 0
    )
    monkeypatch.setattr(
        entrada_train.importlib, "import_module", lambda nombre: types.SimpleNamespace()
    )
    resultado = runner.invoke(entrada_train.app, ["--question", "p3", "--model", "kmeans"])
    assert resultado.exit_code != 0
