"""Pruebas de las funciones de features.py que usa P1 y de la reproducibilidad del entrenamiento.

Cubren casos normales y casos límite de la tabla estación × hora, el calendario, los rezagos,
los promedios, la capacidad y la partición temporal, y verifican que entrenar dos veces con la
misma semilla (config.SEED) da las mismas predicciones y las mismas métricas en MLflow.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici_mlops import features, tracking
from mibici_mlops.config import FECHA_FIN, FECHA_INICIO, SEED, TEST_INICIO, VAL_INICIO
from mibici_mlops.modeling import p1

FIN_GRID = pd.Timestamp(FECHA_FIN) + pd.Timedelta(hours=23)


def viajes(filas):
    """Viajes de ejemplo a partir de tuplas (inicio, fin, origen, destino)."""
    datos = pd.DataFrame(filas, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"])
    datos["inicio_viaje"] = pd.to_datetime(datos["inicio_viaje"])
    datos["fin_viaje"] = pd.to_datetime(datos["fin_viaje"])
    return datos


def serie(valores, estacion=1, inicio="2026-01-05"):
    """Tabla estación × hora con una serie de salidas consecutivas para una estación."""
    return pd.DataFrame(
        {
            "estacion_id": estacion,
            "hora": pd.date_range(inicio, periods=len(valores), freq="h"),
            "salidas": valores,
        }
    )


# ---------------------------------------------------------------- agregar_estacion_hora
def test_estacion_hora_cuenta_salidas_llegadas_y_flujo():
    tabla = features.agregar_estacion_hora(
        viajes(
            [
                ("2026-01-05 08:05", "2026-01-05 08:20", 1, 2),
                ("2026-01-05 08:40", "2026-01-05 08:55", 1, 2),
                ("2026-01-05 08:50", "2026-01-05 09:10", 2, 1),
            ]
        )
    )
    h8 = tabla[tabla["hora"] == "2026-01-05 08:00"].set_index("estacion_id")
    h9 = tabla[tabla["hora"] == "2026-01-05 09:00"].set_index("estacion_id")
    assert h8.loc[1, "salidas"] == 2 and h8.loc[1, "llegadas"] == 0
    assert h8.loc[2, "salidas"] == 1 and h8.loc[2, "llegadas"] == 2
    assert h9.loc[1, "llegadas"] == 1
    assert (tabla["flujo_neto"] == tabla["llegadas"] - tabla["salidas"]).all()
    assert tabla["salidas"].sum() == 3 and tabla["llegadas"].sum() == 3


def test_estacion_hora_viaje_que_cruza_de_hora():
    tabla = features.agregar_estacion_hora(
        viajes([("2026-01-05 08:59", "2026-01-05 09:01", 1, 2)])
    ).set_index(["estacion_id", "hora"])
    assert tabla.loc[(1, pd.Timestamp("2026-01-05 08:00")), "salidas"] == 1
    assert tabla.loc[(2, pd.Timestamp("2026-01-05 09:00")), "llegadas"] == 1
    assert (2, pd.Timestamp("2026-01-05 08:00")) not in tabla.index


def test_estacion_hora_rellena_horas_sin_viajes_con_cero():
    tabla = features.agregar_estacion_hora(
        viajes(
            [
                ("2026-01-05 08:10", "2026-01-05 08:30", 1, 1),
                ("2026-01-05 12:10", "2026-01-05 12:30", 1, 1),
            ]
        )
    )
    horas = tabla["hora"]
    assert horas.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert horas.max() == FIN_GRID
    entre = tabla[(horas > "2026-01-05 08:00") & (horas < "2026-01-05 12:00")]
    assert len(entre) == 3
    assert (entre[["salidas", "llegadas"]] == 0).all().all()


def test_estacion_hora_estacion_entra_desde_su_primer_viaje():
    tabla = features.agregar_estacion_hora(
        viajes(
            [
                ("2025-07-01 06:10", "2025-07-01 06:30", 1, 1),
                ("2026-03-10 15:10", "2026-03-10 15:20", 2, 1),
            ]
        )
    )
    inicio = tabla.groupby("estacion_id")["hora"].min()
    assert inicio[1] == pd.Timestamp("2025-07-01 06:00")
    assert inicio[2] == pd.Timestamp("2026-03-10 15:00")
    assert (
        len(tabla[tabla["estacion_id"] == 2]) == (FIN_GRID - inicio[2]) // pd.Timedelta("1h") + 1
    )


def test_estacion_hora_estacion_que_solo_recibe_llegadas():
    tabla = features.agregar_estacion_hora(
        viajes([("2026-01-05 08:10", "2026-01-05 09:30", 1, 7)])
    )
    destino = tabla[tabla["estacion_id"] == 7]
    assert destino["hora"].min() == pd.Timestamp("2026-01-05 09:00")
    assert destino["salidas"].sum() == 0 and destino["llegadas"].sum() == 1


def test_estacion_hora_acepta_tabla_ya_agregada():
    crudos = viajes(
        [
            ("2026-01-05 08:05", "2026-01-05 08:20", 1, 2),
            ("2026-01-05 10:40", "2026-01-05 10:55", 2, 1),
        ]
    )
    desde_viajes = features.agregar_estacion_hora(crudos)
    agregada = desde_viajes[["estacion_id", "hora", "salidas", "llegadas"]]
    desde_agregada = features.agregar_estacion_hora(agregada)
    pd.testing.assert_frame_equal(desde_viajes, desde_agregada)


def test_estacion_hora_descarta_viajes_fuera_del_rango():
    tabla = features.agregar_estacion_hora(
        viajes(
            [
                ("2025-06-30 23:10", "2025-06-30 23:20", 1, 1),
                ("2026-01-05 08:05", "2026-01-05 08:20", 1, 1),
            ]
        )
    )
    assert tabla["hora"].min() >= pd.Timestamp(FECHA_INICIO)
    assert tabla["salidas"].sum() == 1


# ---------------------------------------------------------------- agregar_calendario
def test_calendario_entre_semana_y_fin_de_semana():
    tabla = pd.DataFrame({"hora": pd.to_datetime(["2026-01-05 07:00", "2026-01-04 23:00"])})
    tabla = features.agregar_calendario(tabla)
    lunes, domingo = tabla.iloc[0], tabla.iloc[1]
    assert (lunes["hora_dia"], lunes["dia_semana"], lunes["es_fin_semana"], lunes["mes"]) == (
        7,
        0,
        0,
        1,
    )
    assert (domingo["hora_dia"], domingo["dia_semana"], domingo["es_fin_semana"]) == (23, 6, 1)


def test_calendario_sabado_es_fin_de_semana():
    tabla = features.agregar_calendario(pd.DataFrame({"hora": pd.to_datetime(["2026-01-03"])}))
    assert tabla.loc[0, "dia_semana"] == 5 and tabla.loc[0, "es_fin_semana"] == 1


# ---------------------------------------------------------------- agregar_rezagos
def test_rezagos_toman_el_valor_k_horas_antes():
    tabla = features.agregar_rezagos(serie(list(range(30))), "salidas", (1, 24))
    assert tabla.loc[29, "salidas_lag_1"] == 28
    assert tabla.loc[29, "salidas_lag_24"] == 5
    assert tabla["salidas_lag_1"].isna().sum() == 1
    assert tabla["salidas_lag_24"].isna().sum() == 24


def test_rezagos_no_se_mezclan_entre_estaciones():
    tabla = pd.concat([serie([5, 6, 7], estacion=1), serie([100, 200, 300], estacion=2)])
    tabla = features.agregar_rezagos(tabla.reset_index(drop=True), "salidas", (1,))
    primera_e2 = tabla[tabla["estacion_id"] == 2].iloc[0]
    assert np.isnan(primera_e2["salidas_lag_1"])


def test_rezagos_no_usan_informacion_del_futuro():
    original = features.agregar_rezagos(serie(list(range(10))), "salidas", (1, 2))
    cambiada = serie(list(range(10)))
    cambiada.loc[6:, "salidas"] = 999
    cambiada = features.agregar_rezagos(cambiada, "salidas", (1, 2))
    cols = ["salidas_lag_1", "salidas_lag_2"]
    pd.testing.assert_frame_equal(original.loc[:6, cols], cambiada.loc[:6, cols])


# ---------------------------------------------------------------- agregar_promedios
def test_media_24h_excluye_la_hora_que_se_predice():
    tabla = features.agregar_promedios(serie([1] * 24 + [100]))
    assert tabla["salidas_media_24h"].iloc[:24].isna().all()
    assert tabla.loc[24, "salidas_media_24h"] == 1.0


def test_media_4sem_promedia_la_misma_hora_de_4_semanas_previas():
    valores = np.zeros(168 * 4 + 1)
    valores[[0, 168, 336, 504]] = [4, 8, 12, 16]
    tabla = features.agregar_promedios(serie(valores))
    assert tabla.loc[672, "salidas_media_4sem"] == 10.0


def test_media_4sem_vacia_hasta_tener_4_semanas_de_historia():
    tabla = features.agregar_promedios(serie(np.ones(168 * 5)))
    assert tabla["salidas_media_4sem"].iloc[:672].isna().all()
    assert tabla["salidas_media_4sem"].iloc[672:].notna().all()


# ---------------------------------------------------------------- agregar_capacidad
def test_capacidad_quita_estaciones_sin_anclajes_o_fuera_del_catalogo():
    tabla = pd.DataFrame({"estacion_id": [1, 2, 3], "hora": pd.Timestamp("2026-01-05")})
    estaciones = pd.DataFrame({"estacion_id": [1, 2], "capacidad": [15, 0]})
    resultado = features.agregar_capacidad(tabla, estaciones)
    assert resultado["estacion_id"].tolist() == [1]
    assert resultado.loc[0, "capacidad"] == 15


# ---------------------------------------------------------------- particion_temporal
def test_particion_respeta_los_cortes_exactos():
    horas = pd.to_datetime(
        [
            FECHA_INICIO,
            pd.Timestamp(VAL_INICIO) - pd.Timedelta(hours=1),
            VAL_INICIO,
            pd.Timestamp(TEST_INICIO) - pd.Timedelta(hours=1),
            TEST_INICIO,
            FIN_GRID,
        ]
    )
    tabla = features.particion_temporal(pd.DataFrame({"hora": horas}))
    assert tabla["conjunto"].tolist() == ["train", "train", "val", "val", "test", "test"]


def test_particion_sin_traslape_y_en_orden():
    tabla = pd.DataFrame({"hora": pd.date_range(FECHA_INICIO, FIN_GRID, freq="h")})
    tabla = features.particion_temporal(tabla)
    limites = tabla.groupby("conjunto")["hora"].agg(["min", "max"])
    assert set(tabla["conjunto"]) == {"train", "val", "test"}
    assert limites.loc["train", "max"] < limites.loc["val", "min"]
    assert limites.loc["val", "max"] < limites.loc["test", "min"]
    assert len(tabla) == tabla["conjunto"].value_counts().sum()


def test_particion_con_otra_columna_de_fecha():
    tabla = pd.DataFrame({"fecha": pd.to_datetime(["2026-01-01", "2026-05-01", "2026-08-01"])})
    tabla = features.particion_temporal(tabla, columna="fecha")
    assert tabla["conjunto"].tolist() == ["train", "val", "test"]


# ---------------------------------------------------------------- reproducibilidad
@pytest.fixture
def tabla_p1(monkeypatch):
    """Tabla de P1 construida con las funciones reales de features.py sobre datos sintéticos."""
    rng = np.random.default_rng(0)
    horas = pd.date_range(FECHA_INICIO, FIN_GRID, freq="h")
    patron = 2 + np.sin(np.arange(len(horas)) * 2 * np.pi / 24)
    conteos = pd.concat(
        pd.DataFrame(
            {
                "estacion_id": e,
                "hora": horas,
                "salidas": rng.poisson(np.clip(patron + e, 0.1, None)),
                "llegadas": rng.poisson(np.clip(patron + e, 0.1, None)),
            }
        )
        for e in (1, 2, 3)
    )
    tabla = features.agregar_estacion_hora(conteos)
    tabla = features.agregar_calendario(tabla)
    tabla = features.agregar_rezagos(tabla, "salidas", (1, 2, 24, 168))
    tabla = features.agregar_promedios(tabla, "salidas")
    tabla["capacidad"] = tabla["estacion_id"] * 10
    tabla = features.particion_temporal(tabla).dropna(subset=[p1.PRED])
    tabla[p1.CARACTERISTICAS] = tabla[p1.CARACTERISTICAS].astype("float32")
    tabla = tabla.reset_index(drop=True)
    monkeypatch.setattr(p1, "construir_tabla", lambda: tabla)
    return tabla


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


@pytest.mark.parametrize("modelo", ["ridge", "hgb"])
def test_entrenamiento_reproducible_mismas_predicciones(tabla_p1, modelo):
    train = tabla_p1[tabla_p1["conjunto"] == "train"]
    test = tabla_p1[tabla_p1["conjunto"] == "test"]
    predicciones = []
    for _ in range(2):
        m = p1.crear_modelo(modelo, {"max_iter": 50} if modelo == "hgb" else {})
        m.fit(train[p1.CARACTERISTICAS], train[p1.OBJETIVO])
        predicciones.append(m.predict(test[p1.CARACTERISTICAS]))
    np.testing.assert_array_equal(predicciones[0], predicciones[1])


def test_hgb_usa_la_semilla_del_proyecto():
    assert p1.crear_modelo("hgb", {}).random_state == SEED


def test_dos_corridas_registran_las_mismas_metricas(mlflow_temporal, tabla_p1):
    corridas = [mlflow.get_run(p1.train("hgb", {"max_iter": 50})) for _ in range(2)]
    a, b = (c.data.metrics for c in corridas)
    for metrica in ("val_mae", "val_rmse", "test_mae", "test_rmse"):
        assert a[metrica] == b[metrica]
    assert corridas[0].info.run_id != corridas[1].info.run_id
    assert all(c.data.params["seed"] == str(SEED) for c in corridas)
