"""Pruebas de candidatos y búsqueda de hiperparámetros P3."""

import pandas as pd
import pytest
from types import SimpleNamespace

import mibici_mlops.modeling.p3 as p3_module
from sklearn.pipeline import Pipeline
from mibici_mlops.modeling.p3_search import (
    K_BUSQUEDA,
    seleccionar_mejor_candidato_p3,
)
from mibici_mlops.modeling.p3_candidates import (
    CANDIDATOS_P3,
    FLUJO_COL,
    crear_candidato_p3,
)


COLUMNAS = [
    "salidas_semana_h00",
    "llegadas_semana_h00",
    FLUJO_COL,
]


def test_cuatro_configuraciones_disponibles():
    assert CANDIDATOS_P3 == (
        "baseline",
        "kmeans-sin-escalar",
        "kmeans-flujo-escalado",
        "pca-flujo-escalado",
    )


@pytest.mark.parametrize("nombre", CANDIDATOS_P3)
def test_candidatos_devuelven_pipeline(nombre):
    modelo = crear_candidato_p3(
        nombre,
        columnas=COLUMNAS,
        params={"k": 3},
    )

    assert isinstance(modelo, Pipeline)
    assert modelo.named_steps["kmeans"].n_clusters == 3
    assert modelo.named_steps["kmeans"].random_state == 42
    assert modelo.named_steps["kmeans"].n_init == 20


def test_baseline_incluye_scaler():
    modelo = crear_candidato_p3(
        "baseline",
        columnas=COLUMNAS,
    )

    assert "scaler" in modelo.named_steps


def test_sin_escalar_no_agrega_preprocesamiento():
    modelo = crear_candidato_p3(
        "kmeans-sin-escalar",
        columnas=COLUMNAS,
    )

    assert list(modelo.named_steps) == ["kmeans"]


def test_flujo_escalado_incluye_preprocesador():
    modelo = crear_candidato_p3(
        "kmeans-flujo-escalado",
        columnas=COLUMNAS,
    )

    assert "preprocesador" in modelo.named_steps
    assert "kmeans" in modelo.named_steps


def test_pca_incluye_reduccion_dimensional():
    modelo = crear_candidato_p3(
        "pca-flujo-escalado",
        columnas=COLUMNAS,
        params={"pca_varianza": 0.90},
    )

    assert "preprocesador" in modelo.named_steps
    assert "pca" in modelo.named_steps
    assert "kmeans" in modelo.named_steps


def test_parametro_k_es_configurable():
    modelo = crear_candidato_p3(
        "kmeans-flujo-escalado",
        columnas=COLUMNAS,
        params={"k": 5},
    )

    assert modelo.named_steps["kmeans"].n_clusters == 5


def test_rechaza_modelo_desconocido():
    with pytest.raises(ValueError, match="Candidato P3 desconocido"):
        crear_candidato_p3(
            "modelo-inexistente",
            columnas=COLUMNAS,
        )


def test_rechaza_k_invalido():
    with pytest.raises(ValueError, match="al menos 2 clusters"):
        crear_candidato_p3(
            "baseline",
            columnas=COLUMNAS,
            params={"k": 1},
        )


def test_rechaza_pca_varianza_invalida():
    with pytest.raises(ValueError, match="pca_varianza"):
        crear_candidato_p3(
            "pca-flujo-escalado",
            columnas=COLUMNAS,
            params={"pca_varianza": 1.5},
        )


def test_rechaza_perfil_sin_flujo():
    columnas = [
        "salidas_semana_h00",
        "llegadas_semana_h00",
    ]

    with pytest.raises(ValueError, match=FLUJO_COL):
        crear_candidato_p3(
            "baseline",
            columnas=columnas,
        )


def test_pipeline_entrena_y_predice():
    X = pd.DataFrame(
        {
            "salidas_semana_h00": [
                0.10,
                0.11,
                0.09,
                0.80,
                0.82,
                0.78,
            ],
            "llegadas_semana_h00": [
                0.12,
                0.10,
                0.11,
                0.75,
                0.80,
                0.77,
            ],
            FLUJO_COL: [
                -0.05,
                -0.04,
                -0.06,
                0.10,
                0.12,
                0.11,
            ],
        }
    )

    modelo = crear_candidato_p3(
        "kmeans-flujo-escalado",
        columnas=list(X.columns),
        params={"k": 2},
    )

    modelo.fit(X)
    predicciones = modelo.predict(X)

    assert len(predicciones) == len(X)
    assert len(set(predicciones)) == 2



def test_busqueda_p3_evalua_seis_valores_de_k():
    assert K_BUSQUEDA == (3, 4, 5, 6, 7, 8)


def test_selecciona_mayor_silhouette():
    resultados = [
        {
            "nombre": "k3",
            "metricas": {
                "val_silhouette": 0.10,
                "val_davies_bouldin": 2.0,
            },
        },
        {
            "nombre": "k4",
            "metricas": {
                "val_silhouette": 0.15,
                "val_davies_bouldin": 2.5,
            },
        },
    ]

    mejor = seleccionar_mejor_candidato_p3(resultados)

    assert mejor["nombre"] == "k4"


def test_davies_bouldin_desempata_silhouette():
    resultados = [
        {
            "nombre": "k3",
            "metricas": {
                "val_silhouette": 0.15,
                "val_davies_bouldin": 2.1,
            },
        },
        {
            "nombre": "k4",
            "metricas": {
                "val_silhouette": 0.15,
                "val_davies_bouldin": 1.8,
            },
        },
    ]

    mejor = seleccionar_mejor_candidato_p3(resultados)

    assert mejor["nombre"] == "k4"


def test_busqueda_p3_vacia_genera_error():
    with pytest.raises(ValueError, match="No hay resultados válidos"):
        seleccionar_mejor_candidato_p3([])

class FakeRun:
    def __init__(self):
        self.info = SimpleNamespace(run_id="run-p3-test")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeModelo:
    def fit(self, X):
        self.X_fit = X
        return self

    def predict(self, X):
        return [0, 1, 0, 1][: len(X)]


def test_train_p3_baseline_integra_flujo(monkeypatch):
    perfil_train = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3, 4],
            "salidas_semana_h00": [0.1, 0.2, 0.8, 0.9],
            FLUJO_COL: [0.1, 0.2, -0.1, -0.2],
        }
    )

    perfil_val = perfil_train.copy()
    perfil_test = perfil_train.copy()
    perfil_final = perfil_train.copy()

    columnas = [
        "salidas_semana_h00",
        FLUJO_COL,
    ]

    monkeypatch.setattr(
        p3_module,
        "preparar_experimento_p3",
        lambda: {
            "viajes": pd.DataFrame(),
            "estaciones": pd.DataFrame(),
            "perfiles": {
                "train": perfil_train,
                "val": perfil_val,
                "test": perfil_test,
            },
            "columnas": columnas,
            "X_train": perfil_train[columnas],
            "X_val": perfil_val[columnas],
            "X_test": perfil_test[columnas],
        },
    )

    monkeypatch.setattr(
        p3_module,
        "iniciar_run",
        lambda **kwargs: FakeRun(),
    )

    monkeypatch.setattr(
        p3_module,
        "crear_candidato_p3",
        lambda *args, **kwargs: FakeModelo(),
    )

    monkeypatch.setattr(
        p3_module,
        "evaluar_candidato_p3",
        lambda *args, **kwargs: {
            "val_silhouette": 0.20,
            "val_davies_bouldin": 1.50,
        },
    )

    monkeypatch.setattr(
        p3_module,
        "construir_perfil_final_p3",
        lambda *args, **kwargs: perfil_final,
    )

    resultado = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3, 4],
            "cluster": [0, 1, 0, 1],
        }
    )

    monkeypatch.setattr(
        p3_module,
        "evaluar_pipeline_final_p3",
        lambda *args, **kwargs: (
            {
                "train_silhouette": 0.30,
                "train_davies_bouldin": 1.20,
                "test_estabilidad_ari": 0.80,
                "test_estaciones_comparables": 4.0,
            },
            resultado,
            resultado,
        ),
    )

    monkeypatch.setattr(
        p3_module,
        "generar_graficas_p3",
        lambda **kwargs: [],
    )

    registrado = {}

    def fake_registrar_resultados(**kwargs):
        registrado.update(kwargs)

    monkeypatch.setattr(
        p3_module,
        "registrar_resultados",
        fake_registrar_resultados,
    )

    run_id = p3_module.train(
        model="baseline",
        params={"k": 2},
    )

    assert run_id == "run-p3-test"

    assert registrado["metricas"]["val_silhouette"] == 0.20
    assert registrado["metricas"]["train_silhouette"] == 0.30
    assert registrado["metricas"]["test_estabilidad_ari"] == 0.80

    assert registrado["nombre_datos"] == (
        "p3-perfil-estacion-train-val"
    )

    assert registrado["config"]["test_usado_para_seleccion"] is False

def test_train_p3_rechaza_modelo_desconocido():
    with pytest.raises(ValueError, match="Modelo P3 desconocido"):
        p3_module.train(
            model="modelo-inexistente",
            params={},
        )
