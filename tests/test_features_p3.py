"""Pruebas unitarias para el feature engineering de P3."""

import pandas as pd
import pytest

from mibici_mlops.features import crear_perfil_p3


def crear_tabla_ejemplo() -> pd.DataFrame:
    """Crea una tabla estación-hora pequeña para probar P3."""
    return pd.DataFrame(
        {
            "estacion_id": [1, 1, 1, 2, 2, 2],
            "hora": pd.to_datetime(
                [
                    "2026-01-05 08:00:00",
                    "2026-01-05 18:00:00",
                    "2026-01-10 12:00:00",
                    "2026-01-05 08:00:00",
                    "2026-01-05 18:00:00",
                    "2026-01-10 12:00:00",
                ]
            ),
            "salidas": [10, 5, 3, 4, 6, 2],
            "llegadas": [5, 10, 6, 8, 2, 4],
            "flujo_neto": [-5, 5, 3, 4, -4, 2],
            "capacidad": [20, 20, 20, 30, 30, 30],
        }
    )


def test_perfil_p3_estructura():
    """Verifica una fila por estación y las 97 features esperadas."""
    perfil = crear_perfil_p3(crear_tabla_ejemplo())

    assert perfil.shape == (2, 98)
    assert perfil["estacion_id"].nunique() == 2
    assert perfil["estacion_id"].duplicated().sum() == 0

    features = [c for c in perfil.columns if c != "estacion_id"]
    assert len(features) == 97


def test_perfil_p3_sin_nulos():
    """Verifica que el perfil no produzca NaN ni infinitos."""
    perfil = crear_perfil_p3(crear_tabla_ejemplo())

    assert perfil.isna().sum().sum() == 0

    numericas = perfil.select_dtypes(include="number")
    assert numericas.isin([float("inf"), float("-inf")]).sum().sum() == 0


def test_perfil_p3_proporciones():
    """Verifica que cada distribución horaria con actividad sume 1."""
    perfil = crear_perfil_p3(crear_tabla_ejemplo())

    prefijos = [
        "salidas_semana_h",
        "llegadas_semana_h",
        "salidas_finde_h",
        "llegadas_finde_h",
    ]

    for prefijo in prefijos:
        columnas = [c for c in perfil.columns if c.startswith(prefijo)]
        suma = perfil[columnas].sum(axis=1)

        assert suma.round(6).eq(1.0).all()


def test_perfil_p3_no_usa_ubicacion():
    """Latitud y longitud no deben formar parte de las features."""
    perfil = crear_perfil_p3(crear_tabla_ejemplo())

    assert "latitud" not in perfil.columns
    assert "longitud" not in perfil.columns
    assert "nombre" not in perfil.columns


def test_perfil_p3_rechaza_capacidad_cero():
    """Una estación con capacidad 0 no puede normalizar el flujo."""
    tabla = crear_tabla_ejemplo()
    tabla.loc[tabla["estacion_id"] == 1, "capacidad"] = 0

    with pytest.raises(ValueError, match="capacidad mayor a 0"):
        crear_perfil_p3(tabla)


def test_perfil_p3_requiere_columnas():
    """Verifica que se detecten columnas obligatorias faltantes."""
    tabla = crear_tabla_ejemplo().drop(columns=["flujo_neto"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        crear_perfil_p3(tabla)