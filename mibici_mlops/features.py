"""Feature engineering del proyecto MiBici (Equipo 42).

Todas las funciones parten de la tabla estación × hora y solo usan información
disponible al momento de la predicción: los rezagos y promedios se calculan con
desplazamientos hacia el pasado sobre una cuadrícula horaria completa.
"""

import pandas as pd

from mibici_mlops.config import FECHA_FIN, FECHA_INICIO, TEST_INICIO, VAL_INICIO


def agregar_estacion_hora(viajes: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla base estación × hora que usan P1, P2 y P3.

    Cuenta las salidas por estación de origen y hora de inicio, y las llegadas por
    estación de destino y hora de fin. Arma la cuadrícula completa del rango del
    proyecto con 0 en las horas sin viajes, porque sin esas filas un shift(k) no
    equivale a k horas atrás. Cada estación entra desde su primer viaje, para no
    llenar con ceros el periodo en que todavía no existía (el sistema pasó de 367
    estaciones con viajes en jul-2025 a 468 en sep-2026).

    Pertinencia: P1 predice las salidas; P2 usa el flujo neto (llegadas − salidas)
    relativo a la capacidad; P3 resume salidas y llegadas por hora en un perfil por
    estación.

    Espera las columnas inicio_viaje, fin_viaje (datetime), origen_id y destino_id.
    Devuelve estacion_id, hora, salidas, llegadas y flujo_neto, ordenada por estación y hora.
    """
    sal = viajes.groupby(["origen_id", viajes["inicio_viaje"].dt.floor("h")]).size()
    lle = viajes.groupby(["destino_id", viajes["fin_viaje"].dt.floor("h")]).size()
    sal.index.names = ["estacion_id", "hora"]
    lle.index.names = ["estacion_id", "hora"]

    estaciones = sorted(set(sal.index.get_level_values(0)) | set(lle.index.get_level_values(0)))
    horas = pd.date_range(
        pd.Timestamp(FECHA_INICIO),
        pd.Timestamp(FECHA_FIN) + pd.Timedelta(hours=23),
        freq="h",
    )
    idx = pd.MultiIndex.from_product([estaciones, horas], names=["estacion_id", "hora"])
    tabla = (
        pd.DataFrame({"salidas": sal, "llegadas": lle})
        .reindex(idx)
        .fillna(0)
        .astype("int32")
        .reset_index()
    )

    activa = (tabla["salidas"] + tabla["llegadas"]) > 0
    primera_hora = tabla.loc[activa].groupby("estacion_id")["hora"].min()
    tabla = tabla[tabla["hora"] >= tabla["estacion_id"].map(primera_hora)]

    tabla["flujo_neto"] = tabla["llegadas"] - tabla["salidas"]
    return tabla.sort_values(["estacion_id", "hora"]).reset_index(drop=True)


def agregar_capacidad(tabla: pd.DataFrame, estaciones: pd.DataFrame) -> pd.DataFrame:
    """Agrega la capacidad (número de anclajes) de cada estación desde la nomenclatura.

    Pertinencia: P2 define el riesgo comparando el flujo neto con la capacidad, y P3
    normaliza el flujo neto por capacidad. Las estaciones con capacidad 0 se quitan
    porque no permiten esa normalización.
    """
    tabla = tabla.merge(estaciones[["estacion_id", "capacidad"]], on="estacion_id", how="left")
    return tabla[tabla["capacidad"] > 0].reset_index(drop=True)


def agregar_calendario(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega hora del día, día de la semana, fin de semana y mes de la hora que se predice.

    Pertinencia: la demanda depende sobre todo de la hora y del tipo de día (en jul-2025,
    unos 13 mil viajes diarios entre semana contra unos 8 mil en fin de semana). Son
    variables que se conocen de antemano, así que no hay fuga temporal. Las usan P1 y P2.
    """
    h = tabla["hora"]
    tabla["hora_dia"] = h.dt.hour
    tabla["dia_semana"] = h.dt.dayofweek
    tabla["es_fin_semana"] = (h.dt.dayofweek >= 5).astype("int8")
    tabla["mes"] = h.dt.month
    return tabla


def agregar_rezagos(
    tabla: pd.DataFrame, columna: str = "salidas", rezagos: tuple = (1, 2, 24, 168)
) -> pd.DataFrame:
    """Agrega el valor de la misma estación k horas antes, para cada k en rezagos.

    Pertinencia: la demanda reciente (1 y 2 h), la del día anterior (24 h) y la de la
    semana anterior a la misma hora (168 h) son los mejores indicadores de la siguiente
    hora. El rezago de 168 h es además el baseline de P1. Para P2 se aplica a flujo_neto.

    Requiere la tabla ordenada y con la cuadrícula completa de agregar_estacion_hora;
    shift(k) solo mira hacia atrás, así que no usa información del futuro.
    """
    g = tabla.groupby("estacion_id")[columna]
    for k in rezagos:
        tabla[f"{columna}_lag_{k}"] = g.shift(k)
    return tabla


def agregar_promedios(tabla: pd.DataFrame, columna: str = "salidas") -> pd.DataFrame:
    """Agrega el promedio de las últimas 24 h y el de la misma hora en las 4 semanas previas.

    Pertinencia: suavizan el ruido de un solo rezago (53 % de las estación-horas de
    jul-2025 tienen 0 salidas) y capturan el nivel típico de la estación a esa hora.
    El primer promedio se desplaza una hora antes de calcularse, así que tampoco usa la
    hora que se predice. Los usan P1 y P2.
    """
    g = tabla.groupby("estacion_id")[columna]
    tabla[f"{columna}_media_24h"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    tabla[f"{columna}_media_4sem"] = sum(g.shift(168 * k) for k in range(1, 5)) / 4
    return tabla


def particion_temporal(tabla: pd.DataFrame, columna: str = "hora") -> pd.DataFrame:
    """Asigna cada fila a train, val o test según su fecha, con los cortes de config.py.

    Pertinencia: la partición es por tiempo para que validación y test sean siempre
    posteriores al entrenamiento y el modelo nunca vea el futuro. La usan P1, P2 y P3
    (en P3, para revisar la estabilidad de los grupos).
    """
    t = tabla[columna]
    tabla["conjunto"] = "train"
    tabla.loc[t >= pd.Timestamp(VAL_INICIO), "conjunto"] = "val"
    tabla.loc[t >= pd.Timestamp(TEST_INICIO), "conjunto"] = "test"
    return tabla

def crear_perfil_p3(tabla: pd.DataFrame) -> pd.DataFrame:
    """Construye el perfil de uso por estación para P3.

    Para cada estación calcula la distribución proporcional de salidas y llegadas
    por hora del día, separando días entre semana y fines de semana.

    Se generan cuatro perfiles de 24 horas:
    - salidas entre semana
    - llegadas entre semana
    - salidas en fin de semana
    - llegadas en fin de semana

    También calcula el flujo neto promedio relativo a la capacidad de la estacion.

    Pertinencia para P3:
    Estas variables describen la forma del patron de uso de cada estacion sin hacer
    que las estaciones con mayor volumen dominen el clustering. La ubicacion
    geografica no se incluye como variable del modelo; se utilizara posteriormente
    para interpretar y visualizar los clusters.

    La funcion debe ejecutarse por separado para train, validacion y test, evitando
    combinar periodos futuros durante la construccion de los perfiles.

    Requiere las columnas:
    estacion_id, hora, salidas, llegadas, flujo_neto y capacidad.

    Devuelve una fila por estación.
    """
    columnas_requeridas = {
        "estacion_id",
        "hora",
        "salidas",
        "llegadas",
        "flujo_neto",
        "capacidad",
    }

    faltantes = columnas_requeridas - set(tabla.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas para P3: {sorted(faltantes)}"
        )

    if tabla["capacidad"].isna().any():
        raise ValueError("P3 no admite estaciones sin capacidad")

    if (tabla["capacidad"] <= 0).any():
        raise ValueError("P3 requiere capacidad mayor a 0")

    datos = tabla.copy()

    datos["hora_dia"] = datos["hora"].dt.hour
    datos["tipo_dia"] = "semana"
    datos.loc[
        datos["hora"].dt.dayofweek >= 5,
        "tipo_dia",
    ] = "finde"

    estaciones = sorted(datos["estacion_id"].unique())
    perfil = pd.DataFrame(index=estaciones)
    perfil.index.name = "estacion_id"

    for tipo_dia in ["semana", "finde"]:
        mascara = datos["tipo_dia"] == tipo_dia

        for movimiento in ["salidas", "llegadas"]:
            agregado = (
                datos.loc[mascara]
                .groupby(["estacion_id", "hora_dia"])[movimiento]
                .sum()
                .unstack(fill_value=0)
                .reindex(columns=range(24), fill_value=0)
            )

            total = agregado.sum(axis=1).replace(0, pd.NA)

            proporciones = (
                agregado.div(total, axis=0)
                .fillna(0.0)
            )

            proporciones.columns = [
                f"{movimiento}_{tipo_dia}_h{hora:02d}"
                for hora in range(24)
            ]

            perfil = perfil.join(proporciones, how="left")

    flujo_relativo = (
        (datos["flujo_neto"] / datos["capacidad"])
        .groupby(datos["estacion_id"])
        .mean()
    )

    perfil["flujo_neto_rel_capacidad"] = flujo_relativo

    return perfil.fillna(0.0).reset_index()