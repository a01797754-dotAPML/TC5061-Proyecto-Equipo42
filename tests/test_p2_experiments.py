"""Pruebas de candidatos y búsqueda de hiperparámetros P2."""

import pytest

from mibici_mlops.modeling.p2_candidates import (
    CANDIDATOS_P2,
    crear_candidato_p2,
)
from mibici_mlops.modeling.p2_search import (
    generar_combinaciones_p2,
    seleccionar_mejor_candidato_p2,
)


def test_tres_candidatos_distintos():
    assert len(CANDIDATOS_P2) == 3

    configuraciones = {
        tuple(sorted(params.items()))
        for params in CANDIDATOS_P2.values()
    }

    assert len(configuraciones) == 3

    for nombre in CANDIDATOS_P2:
        modelo = crear_candidato_p2(nombre)
        assert modelo.random_state == 42


def test_ocho_combinaciones_unicas():
    combinaciones = generar_combinaciones_p2()

    assert len(combinaciones) == 8
    assert len({
        tuple(sorted(params.items()))
        for params in combinaciones
    }) == 8


def test_selecciona_mayor_f1_macro():
    resultados = [
        {
            "nombre": "modelo_a",
            "metricas": {
                "val_macro_f1": 0.50,
                "val_recall_desabasto": 0.40,
                "val_recall_saturacion": 0.45,
            },
        },
        {
            "nombre": "modelo_b",
            "metricas": {
                "val_macro_f1": 0.52,
                "val_recall_desabasto": 0.35,
                "val_recall_saturacion": 0.42,
            },
        },
    ]

    mejor = seleccionar_mejor_candidato_p2(resultados)

    assert mejor["nombre"] == "modelo_b"


def test_busqueda_vacia_genera_error():
    with pytest.raises(ValueError):
        seleccionar_mejor_candidato_p2([])
