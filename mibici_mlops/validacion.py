"""Validación de datos con DeepChecks: integridad de los datos procesados y drift train/test.

Corre en el entorno aparte de requirements-deepchecks.txt (DeepChecks 0.19.1 no es compatible
con numpy 2 ni con scikit-learn 1.8+). Los reportes HTML se guardan en reports/deepchecks/ y un
resumen de las condiciones en reports/deepchecks/resumen.json.
"""

import argparse
import json

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import PercentOfNulls
from deepchecks.tabular.suites import data_integrity, train_test_validation
from loguru import logger
import pandas as pd

from mibici_mlops.config import (
    ESTACIONES_PATH,
    INTERIM_DATA_DIR,
    REPORTS_DIR,
    SEED,
    VIAJES_PATH,
)

DEEPCHECKS_DIR = REPORTS_DIR / "deepchecks"
P1_TABLA = INTERIM_DATA_DIR / "p1_tabla.parquet"
MUESTRA = 200_000
MAX_NULOS = 0.05


def muestra(tabla: pd.DataFrame, n: int = MUESTRA) -> pd.DataFrame:
    """Muestra con semilla fija para que el reporte sea reproducible."""
    if len(tabla) <= n:
        return tabla
    return tabla.sample(n=n, random_state=SEED).sort_index()


def cargar_viajes() -> pd.DataFrame:
    archivos = sorted(VIAJES_PATH.glob("viajes_*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No hay viajes en {VIAJES_PATH}; corre `make data` primero")
    viajes = pd.concat((pd.read_parquet(a) for a in archivos), ignore_index=True)
    logger.info(f"Viajes: {len(viajes):,} filas en {len(archivos)} archivos")
    viajes["duracion_min"] = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds() / 60
    return viajes


def suite_integridad():
    """data_integrity más el porcentaje de nulos por columna, que la suite no trae."""
    suite = data_integrity()
    suite.add(PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(MAX_NULOS))
    return suite


def reglas_negocio(viajes: pd.DataFrame, estaciones: pd.DataFrame) -> dict:
    """Conteos sobre todos los datos (sin muestra) que DeepChecks no revisa por sí solo."""
    catalogo = set(estaciones["estacion_id"])
    return {
        "viajes": len(viajes),
        "viajes_id_duplicado": int(viajes["viaje_id"].duplicated().sum()),
        "viajes_filas_duplicadas": int(viajes.duplicated().sum()),
        "viajes_nulos": int(viajes.isna().sum().sum()),
        "viajes_duracion_no_positiva": int((viajes["duracion_min"] <= 0).sum()),
        "viajes_duracion_mayor_24h": int((viajes["duracion_min"] > 24 * 60).sum()),
        "viajes_origen_fuera_catalogo": int((~viajes["origen_id"].isin(catalogo)).sum()),
        "viajes_destino_fuera_catalogo": int((~viajes["destino_id"].isin(catalogo)).sum()),
        "estaciones": len(estaciones),
        "estaciones_capacidad_no_positiva": int((estaciones["capacidad"] <= 0).sum()),
    }


def dataset_viajes(viajes: pd.DataFrame) -> Dataset:
    return Dataset(
        muestra(viajes),
        label=None,
        cat_features=["origen_id", "destino_id"],
        datetime_name="inicio_viaje",
        index_name="viaje_id",
    )


def dataset_estaciones(estaciones: pd.DataFrame) -> Dataset:
    return Dataset(estaciones, label=None, cat_features=[], index_name="estacion_id")


def cargar_p1() -> pd.DataFrame:
    if not P1_TABLA.exists():
        raise FileNotFoundError(f"No existe {P1_TABLA}; corre el entrenamiento de P1 primero")
    return pd.read_parquet(P1_TABLA)


def datasets_p1(tabla: pd.DataFrame) -> tuple[Dataset, Dataset]:
    columnas = [c for c in tabla.columns if c != "conjunto"]

    def crear(conjunto: str) -> Dataset:
        datos = muestra(tabla.loc[tabla["conjunto"] == conjunto, columnas])
        return Dataset(datos, label="salidas", cat_features=["estacion_id"], datetime_name="hora")

    return crear("train"), crear("test")


def resumir(resultado) -> dict:
    """Condiciones aprobadas, no aprobadas y chequeos que no corrieron."""
    return {
        "aprobados": [r.get_header() for r in resultado.get_passed_checks()],
        "no_aprobados": [r.get_header() for r in resultado.get_not_passed_checks()],
        "no_corrieron": [r.get_header() for r in resultado.get_not_ran_checks()],
    }


def guardar(resultado, nombre: str) -> None:
    """Guarda el reporte HTML; DeepChecks no sobrescribe y agregaría " (1)" al nombre."""
    archivo = DEEPCHECKS_DIR / f"{nombre}.html"
    archivo.unlink(missing_ok=True)
    resultado.save_as_html(str(archivo), as_widget=False)


def main(solo: str | None = None) -> dict:
    DEEPCHECKS_DIR.mkdir(parents=True, exist_ok=True)
    archivo_resumen = DEEPCHECKS_DIR / "resumen.json"
    # Con --solo se conserva lo que ya había de la otra parte
    resumen = json.loads(archivo_resumen.read_text()) if solo and archivo_resumen.exists() else {}

    tabla_p1 = cargar_p1()
    train, test = datasets_p1(tabla_p1)

    if solo in (None, "integridad"):
        viajes = cargar_viajes()
        estaciones = pd.read_parquet(ESTACIONES_PATH)
        datasets = {
            "viajes": dataset_viajes(viajes),
            "estaciones": dataset_estaciones(estaciones),
            "p1_train": train,
        }
        for nombre, dataset in datasets.items():
            logger.info(f"DeepChecks: integridad de {nombre}")
            resultado = suite_integridad().run(dataset)
            guardar(resultado, f"integridad_{nombre}")
            resumen[f"integridad_{nombre}"] = resumir(resultado)
        resumen["reglas_negocio"] = reglas_negocio(viajes, estaciones)

    if solo in (None, "drift"):
        logger.info("DeepChecks: drift entre train y test de P1")
        resultado = train_test_validation().run(train_dataset=train, test_dataset=test)
        guardar(resultado, "drift_p1")
        resumen["drift_p1"] = resumir(resultado)

    archivo_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    for suite, partes in resumen.items():
        if suite == "reglas_negocio":
            logger.info(f"Reglas de negocio: {partes}")
            continue
        logger.info(
            f"{suite}: {len(partes['aprobados'])} aprobados, {len(partes['no_aprobados'])} no aprobados"
        )
        for check in partes["no_aprobados"]:
            logger.warning(f"  {suite} · no aprobado: {check}")
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo", choices=["integridad", "drift"], default=None)
    main(parser.parse_args().solo)
