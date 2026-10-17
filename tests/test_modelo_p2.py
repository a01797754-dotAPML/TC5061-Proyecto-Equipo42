"""Pruebas unitarias para el modelo P2 de MiBici."""

from mibici_mlops.modelo_p2 import (
    OBJETIVO_P2,
    VARIABLES_P2,
    crear_modelo_p2,
)


def test_variables_predictoras_p2():
    """Verifica las variables predictoras y la etiqueta."""
    assert len(VARIABLES_P2) == 11
    assert len(set(VARIABLES_P2)) == 11
    assert OBJETIVO_P2 == "riesgo_p2_t_3h"

    # Evitar fuga de información de la hora actual.
    assert "salidas" not in VARIABLES_P2
    assert "llegadas" not in VARIABLES_P2
    assert "flujo_neto" not in VARIABLES_P2
    assert "flujo_relativo" not in VARIABLES_P2


def test_configuracion_random_forest():
    """Verifica los parámetros seleccionados para P2."""
    modelo = crear_modelo_p2()

    assert modelo.n_estimators == 50
    assert modelo.max_depth == 12
    assert modelo.min_samples_leaf == 20
    assert modelo.class_weight == {0: 1, 1: 10, 2: 10}
    assert modelo.random_state == 42
    assert modelo.n_jobs == 2


def test_guardar_y_cargar_modelo_p2(tmp_path):
    """Comprueba el guardado y la recuperación de un modelo."""
    import joblib

    modelo = crear_modelo_p2()
    ruta = tmp_path / "modelo_p2.joblib"

    joblib.dump(modelo, ruta)
    recuperado = joblib.load(ruta)

    assert recuperado.n_estimators == 50
    assert recuperado.class_weight == {0: 1, 1: 10, 2: 10}
