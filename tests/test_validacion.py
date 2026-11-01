"""Pruebas de la validación de datos con DeepChecks.

Solo corren en el entorno de requirements-deepchecks.txt; en el entorno principal se saltan.
"""

import pandas as pd
import pytest

pytest.importorskip("deepchecks")

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates, NewCategoryTrainTest

from mibici_mlops import validacion


def crear_viajes() -> pd.DataFrame:
    """Viajes de ejemplo con una duración cero, una mayor a 24 h y una estación fuera del catálogo."""
    inicio = pd.to_datetime(["2026-01-05 08:00", "2026-01-05 09:00", "2026-01-05 10:00"])
    fin = pd.to_datetime(["2026-01-05 08:15", "2026-01-05 09:00", "2026-01-06 11:00"])
    viajes = pd.DataFrame(
        {
            "viaje_id": [1, 2, 3],
            "inicio_viaje": inicio,
            "fin_viaje": fin,
            "origen_id": [10, 11, 99],
            "destino_id": [11, 10, 10],
        }
    )
    viajes["duracion_min"] = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds() / 60
    return viajes


def test_muestra_es_reproducible():
    tabla = pd.DataFrame({"x": range(1000)})
    a = validacion.muestra(tabla, n=100)
    b = validacion.muestra(tabla, n=100)
    assert len(a) == 100
    pd.testing.assert_frame_equal(a, b)


def test_muestra_tabla_chica_regresa_completa():
    tabla = pd.DataFrame({"x": range(10)})
    assert validacion.muestra(tabla, n=100) is tabla


def test_reglas_negocio_detecta_problemas():
    estaciones = pd.DataFrame({"estacion_id": [10, 11], "capacidad": [20, 0]})
    reglas = validacion.reglas_negocio(crear_viajes(), estaciones)
    assert reglas["viajes_duracion_no_positiva"] == 1
    assert reglas["viajes_duracion_mayor_24h"] == 1
    assert reglas["viajes_origen_fuera_catalogo"] == 1
    assert reglas["viajes_destino_fuera_catalogo"] == 0
    assert reglas["estaciones_capacidad_no_positiva"] == 1
    assert reglas["viajes_id_duplicado"] == 0


def test_reglas_negocio_sin_problemas():
    viajes = crear_viajes().iloc[:1]
    estaciones = pd.DataFrame({"estacion_id": [10, 11], "capacidad": [20, 15]})
    reglas = validacion.reglas_negocio(viajes, estaciones)
    assert all(v == 0 for k, v in reglas.items() if k not in ("viajes", "estaciones"))


def test_suite_integridad_detecta_nulos():
    datos = pd.DataFrame({"a": [1.0, None, None, 4.0] * 25, "b": range(100)})
    resultado = validacion.suite_integridad().run(Dataset(datos, label=None, cat_features=[]))
    assert "PercentOfNulls" in validacion.resumir(resultado)["no_aprobados"]


def test_chequeo_duplicados_detecta_filas_repetidas():
    datos = pd.DataFrame({"a": [1, 1, 2, 3], "b": [5, 5, 6, 7]})
    check = DataDuplicates().add_condition_ratio_less_or_equal(0)
    resultado = check.run(Dataset(datos, label=None, cat_features=[]))
    assert not resultado.passed_conditions()


def test_drift_detecta_categoria_nueva_en_test():
    tabla = pd.DataFrame(
        {
            "estacion_id": [1, 2] * 50 + [1, 3] * 50,
            "hora": pd.date_range("2026-01-01", periods=200, freq="h"),
            "conjunto": ["train"] * 100 + ["test"] * 100,
            "salidas": [1, 2] * 100,
        }
    )
    train, test = validacion.datasets_p1(tabla)
    check = NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(0)
    resultado = check.run(train, test)
    assert not resultado.passed_conditions()
    assert resultado.value.loc["estacion_id", "# New Categories"] == 1
