"""Preparación de datos para el modelo P2 de MiBici."""

from pathlib import Path

import pandas as pd

from mibici_mlops.config import PROJ_ROOT
from mibici_mlops.features import (
    agregar_calendario,
    agregar_capacidad,
    agregar_promedios,
    agregar_rezagos,
    crear_etiqueta_p2,
    particion_temporal,
)


def consolidar_flujos_horarios() -> pd.DataFrame:
    """Consolida salidas y llegadas por estación y hora de todos los meses."""

    carpeta = Path(PROJ_ROOT) / "data" / "processed" / "viajes"
    archivos = sorted(carpeta.glob("viajes_*.parquet"))

    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos en {carpeta}")

    acumulados = []

    for archivo in archivos:
        print(f"Procesando: {archivo.name}")

        viajes = pd.read_parquet(
            archivo,
            columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"],
        )

        salidas = (
            viajes.groupby(["origen_id", viajes["inicio_viaje"].dt.floor("h")])
            .size()
            .rename("salidas")
        )
        salidas.index.names = ["estacion_id", "hora"]

        llegadas = (
            viajes.groupby(["destino_id", viajes["fin_viaje"].dt.floor("h")])
            .size()
            .rename("llegadas")
        )
        llegadas.index.names = ["estacion_id", "hora"]

        mensual = pd.concat([salidas, llegadas], axis=1).fillna(0)
        acumulados.append(mensual.astype("int32"))

        del viajes

    tabla = pd.concat(acumulados)

    tabla = tabla.groupby(level=["estacion_id", "hora"]).sum().reset_index()

    tabla["flujo_neto"] = tabla["llegadas"] - tabla["salidas"]

    print(f"Meses procesados: {len(archivos)}")
    print(f"Registros estación-hora con movimientos: {len(tabla):,}")

    return tabla


def completar_horas_sin_movimientos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Completa las horas sin viajes para cada estación."""

    resultados = []

    for estacion_id, grupo in tabla.groupby("estacion_id"):
        grupo = grupo.sort_values("hora").set_index("hora")

        horas = pd.date_range(
            start=grupo.index.min(),
            end=grupo.index.max(),
            freq="h",
        )

        completo = grupo.reindex(horas)
        completo["estacion_id"] = estacion_id
        completo[["salidas", "llegadas"]] = completo[["salidas", "llegadas"]].fillna(0)

        completo["flujo_neto"] = completo["llegadas"] - completo["salidas"]

        completo.index.name = "hora"
        resultados.append(completo.reset_index())

    resultado = pd.concat(resultados, ignore_index=True)

    return resultado.sort_values(["estacion_id", "hora"]).reset_index(drop=True)


def preparar_base_p2() -> pd.DataFrame:
    """Prepara los flujos horarios con capacidad válida."""

    tabla = consolidar_flujos_horarios()

    estaciones = pd.read_parquet(Path(PROJ_ROOT) / "data" / "processed" / "estaciones.parquet")

    estaciones = estaciones.loc[estaciones["capacidad"] > 0].copy()

    tabla = tabla.loc[tabla["estacion_id"].isin(estaciones["estacion_id"])].copy()

    tabla = completar_horas_sin_movimientos(tabla)
    tabla = agregar_capacidad(tabla, estaciones)

    # Excluir registros posteriores al periodo del proyecto
    tabla = tabla.loc[tabla["hora"] < pd.Timestamp("2026-10-01")].copy()

    return tabla


def generar_dataset_p2() -> pd.DataFrame:
    """Genera el dataset P2 con variables históricas y etiqueta."""

    print("Preparando base P2...")
    tabla = preparar_base_p2()

    print("Agregando variables de calendario...")
    tabla = agregar_calendario(tabla)

    print("Calculando rezagos históricos...")
    tabla = agregar_rezagos(
        tabla,
        columna="salidas",
        rezagos=(1, 2, 24, 168),
    )

    print("Calculando promedios históricos...")
    tabla = agregar_promedios(
        tabla,
        columna="salidas",
    )

    print("Generando etiquetas P2...")
    tabla = crear_etiqueta_p2(
        tabla,
        umbral=0.30,
    )

    # Riesgo de desabasto o saturación tres horas después
    tabla["riesgo_p2_t_3h"] = tabla.groupby("estacion_id")["riesgo_p2"].shift(-3)

    print("Calculando baseline estacional P2 (165 horas)...")
    tabla = agregar_rezagos(
        tabla,
        columna="riesgo_p2",
        rezagos=(165,),
    )

    print("Asignando particiones temporales...")
    tabla = particion_temporal(
        tabla,
        columna="hora",
    )

    return tabla
