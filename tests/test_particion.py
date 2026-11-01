"""Pruebas unitarias de la partición temporal."""

import pandas as pd

from mibici_mlops.config import FECHA_INICIO, TEST_INICIO, VAL_INICIO
from mibici_mlops.features import particion_temporal

UNA_HORA = pd.Timedelta(hours=1)


def test_los_cortes_exactos_entran_al_conjunto_siguiente():
    val, test = pd.Timestamp(VAL_INICIO), pd.Timestamp(TEST_INICIO)
    tabla = pd.DataFrame({"hora": [val - UNA_HORA, val, test - UNA_HORA, test]})
    resultado = particion_temporal(tabla)
    assert resultado["conjunto"].tolist() == ["train", "val", "val", "test"]


def test_los_conjuntos_no_se_traslapan_y_respetan_el_orden():
    horas = pd.date_range(FECHA_INICIO, "2026-09-30 23:00", freq="h")
    resultado = particion_temporal(pd.DataFrame({"hora": horas}))
    limites = resultado.groupby("conjunto")["hora"].agg(["min", "max"])
    assert limites.loc["train", "max"] < limites.loc["val", "min"]
    assert limites.loc["val", "max"] < limites.loc["test", "min"]
    assert len(resultado) == len(horas)


def test_acepta_otra_columna_de_fecha():
    tabla = pd.DataFrame({"inicio": [pd.Timestamp(TEST_INICIO)]})
    assert particion_temporal(tabla, columna="inicio")["conjunto"].tolist() == ["test"]


def test_tabla_vacia_regresa_vacia():
    tabla = pd.DataFrame({"hora": pd.to_datetime([])})
    resultado = particion_temporal(tabla)
    assert resultado.empty
    assert "conjunto" in resultado.columns
