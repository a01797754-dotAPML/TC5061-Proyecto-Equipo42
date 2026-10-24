"""Pruebas de los candidatos de P1 (entrega 2) con una tabla sintética y MLflow temporal."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici_mlops import tracking
from mibici_mlops.modeling import p1


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def tabla_sintetica(monkeypatch):
    rng = np.random.default_rng(0)
    horas = pd.date_range("2026-01-01", periods=24 * 21, freq="h")
    filas = []
    for estacion in (1, 2, 3):
        base = 2 + np.sin(np.arange(len(horas)) * 2 * np.pi / 24) + estacion
        salidas = rng.poisson(np.clip(base, 0.1, None))
        d = pd.DataFrame({"estacion_id": estacion, "hora": horas, "salidas": salidas})
        for k in (1, 2, 24, 168):
            d[f"salidas_lag_{k}"] = d["salidas"].shift(k)
        d["salidas_media_24h"] = d["salidas"].shift(1).rolling(24).mean()
        d["salidas_media_4sem"] = d["salidas_lag_168"]
        filas.append(d)
    tabla = pd.concat(filas).dropna(subset=["salidas_lag_168"]).reset_index(drop=True)
    tabla["hora_dia"] = tabla["hora"].dt.hour
    tabla["dia_semana"] = tabla["hora"].dt.dayofweek
    tabla["es_fin_semana"] = (tabla["dia_semana"] >= 5).astype(int)
    tabla["mes"] = tabla["hora"].dt.month
    tabla["capacidad"] = tabla["estacion_id"] * 10.0
    tabla["conjunto"] = "train"
    tabla.loc[tabla["hora"] >= horas[24 * 14], "conjunto"] = "val"
    tabla.loc[tabla["hora"] >= horas[24 * 17], "conjunto"] = "test"
    tabla[p1.CARACTERISTICAS] = tabla[p1.CARACTERISTICAS].astype("float32")
    monkeypatch.setattr(p1, "construir_tabla", lambda: tabla)
    return tabla


def _run(run_id):
    return mlflow.get_run(run_id)


def test_baseline_y_candidatos(mlflow_temporal, tabla_sintetica):
    base = _run(p1.train("baseline", {}))
    assert base.data.tags["stage_of_work"] == "baseline"
    assert {"val_mae", "test_mae", "val_rmse", "test_rmse"} <= set(base.data.metrics)

    for modelo, params in (
        ("ridge", {"alpha": 2.0}),
        ("hgb", {"max_iter": 20}),
        ("hgb-squared", {"max_iter": 20}),
    ):
        r = _run(p1.train(modelo, params))
        assert r.data.tags["stage_of_work"] == "candidate"
        if modelo == "hgb-squared":
            assert r.info.run_name == "p1-hgb-squared_error"
        assert r.data.tags["question"] == "P1"
        assert {"val_mae", "test_mae", "train_segundos"} <= set(r.data.metrics)
        artefactos = {a.path for a in mlflow.artifacts.list_artifacts(run_id=r.info.run_id)}
        assert {"plots", "config"} <= artefactos
        assert r.inputs.dataset_inputs
        cargado = mlflow.pyfunc.load_model(r.data.tags["model_uri"])
        assert cargado.metadata.signature is not None
        assert len(cargado.predict(tabla_sintetica[p1.CARACTERISTICAS].head(3))) == 3


def test_busqueda_nested_y_final(mlflow_temporal, tabla_sintetica):
    params = {"max_iter": 20, "learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 15]}
    padre = _run(p1.train("hgb-search", params))
    assert padre.data.tags["stage_of_work"] == "search"
    hijos = mlflow.search_runs(
        experiment_ids=[padre.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{padre.info.run_id}'",
        output_format="list",
    )
    assert len(hijos) == 4
    assert padre.data.tags["mejor_hijo_run_id"] in {h.info.run_id for h in hijos}

    final = _run(p1.train("hgb-final", {}))
    assert final.data.params["entrenado_con"] == "train+val"
    assert final.data.tags["busqueda_origen_run_id"] == padre.info.run_id
    assert final.data.params["max_iter"] == "20"
    assert "test_mae" in final.data.metrics and "val_mae" not in final.data.metrics


def test_modelo_desconocido(tabla_sintetica):
    with pytest.raises(ValueError):
        p1.train("xgboost", {})
