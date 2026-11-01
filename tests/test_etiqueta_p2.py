"""Pruebas unitarias de la etiqueta de riesgo de P2."""

import pandas as pd
import pytest

from mibici_mlops.features import agregar_capacidad, crear_etiqueta_p2


def crear_tabla(flujo_neto, capacidad=10) -> pd.DataFrame:
    return pd.DataFrame({"flujo_neto": flujo_neto, "capacidad": capacidad})


def test_asigna_las_tres_clases():
    resultado = crear_etiqueta_p2(crear_tabla([0, -5, 5]))
    assert resultado["riesgo_p2"].tolist() == [0, 1, 2]
    assert resultado["flujo_relativo"].tolist() == [0.0, -0.5, 0.5]


def test_el_umbral_exacto_es_riesgo_bajo():
    """La comparación es estricta: justo en ±30 % todavía no hay riesgo."""
    resultado = crear_etiqueta_p2(crear_tabla([-3, 3]))
    assert resultado["riesgo_p2"].tolist() == [0, 0]


def test_respeta_el_umbral_recibido():
    resultado = crear_etiqueta_p2(crear_tabla([-2, 2]), umbral=0.1)
    assert resultado["riesgo_p2"].tolist() == [1, 2]


def test_no_modifica_la_tabla_original():
    tabla = crear_tabla([1, 2])
    crear_etiqueta_p2(tabla)
    assert list(tabla.columns) == ["flujo_neto", "capacidad"]


def test_tabla_vacia_regresa_vacia():
    resultado = crear_etiqueta_p2(crear_tabla(pd.Series(dtype="float64")))
    assert resultado.empty
    assert "riesgo_p2" in resultado.columns


def test_capacidad_faltante_queda_como_riesgo_bajo():
    """Sin capacidad no se puede calcular el flujo relativo y no se marca riesgo."""
    resultado = crear_etiqueta_p2(crear_tabla([5], capacidad=float("nan")))
    assert resultado["riesgo_p2"].tolist() == [0]
    assert resultado["flujo_relativo"].isna().all()


def test_estaciones_sin_capacidad_se_quitan_antes_de_etiquetar():
    """agregar_capacidad evita la división entre cero al calcular la etiqueta."""
    tabla = pd.DataFrame({"estacion_id": [1, 2, 3], "flujo_neto": [5, 5, 5]})
    estaciones = pd.DataFrame({"estacion_id": [1, 2], "capacidad": [10, 0]})
    resultado = crear_etiqueta_p2(agregar_capacidad(tabla, estaciones))
    assert resultado["estacion_id"].tolist() == [1]
    assert resultado["flujo_relativo"].tolist() == pytest.approx([0.5])
