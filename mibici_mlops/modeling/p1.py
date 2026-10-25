"""P1 – Regresión: viajes que inician por estación en la siguiente hora.

Baseline naive estacional: la predicción para la estación s a la hora h es el número de
salidas de esa estación a la misma hora una semana antes (rezago de 168 h).

Entrega 1 (baseline):  python -m mibici_mlops.modeling.p1
Entrega 2 (candidatos, desde el MLflow Project):
    python -m mlflow run . -e train --experiment-name mibici-equipo42 -P question=p1 -P model=<modelo>
    con model = baseline | ridge | hgb | hgb-squared | hgb-search | hgb-final (ver train()).
"""

import itertools
import json
import os
from pathlib import Path
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici_mlops import tracking
from mibici_mlops.config import (
    ESTACIONES_PATH,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MLFLOW_EXPERIMENT,
    SEED,
    VIAJES_PATH,
)
from mibici_mlops.features import (
    agregar_calendario,
    agregar_estacion_hora,
    agregar_promedios,
    agregar_rezagos,
    particion_temporal,
)
from mibici_mlops.tracking import (
    iniciar_run,
    obtener_data_version,
    registrar_resultados,
    registrar_run,
)

REZAGO = 168
RUN_NAME = "p1-baseline-semana-anterior"
OBJETIVO = "salidas"
PRED = f"{OBJETIVO}_lag_{REZAGO}"


def cargar_viajes() -> pd.DataFrame:
    return pd.read_parquet(
        VIAJES_PATH, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    )


def evaluar(df: pd.DataFrame) -> dict:
    metricas = {}
    for conjunto in ["val", "test"]:
        d = df[df["conjunto"] == conjunto]
        metricas[f"{conjunto}_mae"] = mean_absolute_error(d[OBJETIVO], d[PRED])
        metricas[f"{conjunto}_rmse"] = root_mean_squared_error(d[OBJETIVO], d[PRED])
    return metricas


def graficar(df: pd.DataFrame, ruta) -> None:
    test = df[df["conjunto"] == "test"]
    total = test.groupby("hora")[[OBJETIVO, PRED]].sum().iloc[: 24 * 14]
    error_hora = (test[OBJETIVO] - test[PRED]).abs().groupby(test["hora"].dt.hour).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    total.plot(ax=axes[0])
    axes[0].set_title("Test: viajes totales por hora (primeras 2 semanas)")
    axes[0].legend(["real", "baseline (semana anterior)"])
    axes[0].set_xlabel("")
    error_hora.plot.bar(ax=axes[1])
    axes[1].set_title("Test: MAE por hora del día")
    axes[1].set_xlabel("hora")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main(run_name: str = RUN_NAME) -> str:
    logger.info("P1: cargando viajes y construyendo la tabla estación-hora")
    tabla = agregar_estacion_hora(cargar_viajes())
    tabla = agregar_rezagos(tabla, OBJETIVO, (REZAGO,))
    tabla = particion_temporal(tabla)

    evaluable = tabla[tabla["conjunto"].isin(["val", "test"])].dropna(subset=[PRED])
    metricas = evaluar(evaluable)
    logger.info(f"P1 métricas: {metricas}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_fig = FIGURES_DIR / "p1_baseline_real_vs_pred.png"
    graficar(evaluable, ruta_fig)

    params = {
        "modelo": "naive_semana_anterior",
        "rezago_horas": REZAGO,
        "objetivo": "salidas_siguiente_hora",
        "granularidad": "estacion-hora",
        "horizonte_horas": 1,
        "n_estaciones": tabla["estacion_id"].nunique(),
        "n_filas_val": int((evaluable["conjunto"] == "val").sum()),
        "n_filas_test": int((evaluable["conjunto"] == "test").sum()),
    }
    return registrar_run("P1", run_name, params, metricas, artefactos=[ruta_fig])


# ---------------------------------------------------------------- entrega 2: candidatos con `mlflow run`
MODELOS = ("baseline", "ridge", "hgb", "hgb-squared", "hgb-search", "hgb-final")
FAMILIAS = {"baseline": "naive_estacional", "ridge": "ridge", "hgb": "hgb"}
CARACTERISTICAS = [
    "hora_dia",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "capacidad",
    "salidas_lag_1",
    "salidas_lag_2",
    "salidas_lag_24",
    "salidas_lag_168",
    "salidas_media_24h",
    "salidas_media_4sem",
]
CATEGORICAS = ["hora_dia", "dia_semana"]
HGB_DEFAULTS = {
    "loss": "poisson",
    "learning_rate": 0.1,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 0.0,
}
BUSQUEDA_DEFAULT = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}
TABLA_CACHE = INTERIM_DATA_DIR / "p1_tabla.parquet"
TIPOS_CONFIABLES = {
    "ridge": ["numpy.dtype"],
    "hgb": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def construir_tabla() -> pd.DataFrame:
    """Tabla estación × hora con las variables de P1, ya particionada.

    Solo conserva las filas con rezago de 168 h, las mismas que evaluó el baseline de la
    entrega 1, para que todas las corridas se comparen sobre exactamente las mismas filas.
    Se guarda en data/interim ligada al data_version para no reconstruirla en cada corrida.
    """
    version = obtener_data_version()
    meta = TABLA_CACHE.with_suffix(".json")
    if (
        TABLA_CACHE.exists()
        and meta.exists()
        and json.loads(meta.read_text()).get("data_version") == version
    ):
        logger.info(f"P1: reutilizando tabla en caché ({version})")
        return pd.read_parquet(TABLA_CACHE)

    logger.info("P1: construyendo la tabla estación-hora con sus variables")
    tabla = agregar_estacion_hora(cargar_viajes())
    tabla = agregar_calendario(tabla)
    tabla = agregar_rezagos(tabla, OBJETIVO, (1, 2, 24, REZAGO))
    tabla = agregar_promedios(tabla, OBJETIVO)
    estaciones = pd.read_parquet(ESTACIONES_PATH, columns=["estacion_id", "capacidad"])
    tabla = tabla.merge(estaciones.drop_duplicates("estacion_id"), on="estacion_id", how="left")
    tabla = particion_temporal(tabla).dropna(subset=[PRED])
    tabla[CARACTERISTICAS] = tabla[CARACTERISTICAS].astype("float32")
    tabla = tabla[["estacion_id", "hora", "conjunto", OBJETIVO, *CARACTERISTICAS]].reset_index(
        drop=True
    )

    TABLA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(TABLA_CACHE, index=False)
    meta.write_text(json.dumps({"data_version": version}))
    return tabla


def crear_modelo(nombre: str, params: dict):
    if nombre == "ridge":
        numericas = [c for c in CARACTERISTICAS if c not in CATEGORICAS]
        preparacion = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                (
                    "num",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    numericas,
                ),
            ]
        )
        return make_pipeline(preparacion, Ridge(alpha=params.get("alpha", 1.0)))
    if nombre == "hgb":
        return HistGradientBoostingRegressor(
            **{**HGB_DEFAULTS, **params}, early_stopping=False, random_state=SEED
        )
    raise ValueError(f"Modelo desconocido para P1: {nombre}. Opciones: {MODELOS}")


def calcular_metricas(y, pred, conjunto: str) -> dict:
    return {
        f"{conjunto}_mae": float(mean_absolute_error(y, pred)),
        f"{conjunto}_rmse": float(root_mean_squared_error(y, pred)),
    }


def graficar_resultados(d: pd.DataFrame, pred, conjunto: str, carpeta: Path) -> list[Path]:
    """Real contra predicho y residuales en el conjunto con que se evalúa la corrida."""
    d = d.assign(pred=pred, residuo=d[OBJETIVO] - pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    total = d.groupby("hora")[[OBJETIVO, "pred"]].sum().iloc[: 24 * 14]
    total.plot(ax=axes[0])
    axes[0].set_title(f"{conjunto}: viajes totales por hora (primeras 2 semanas)")
    axes[0].legend(["real", "predicho"])
    axes[0].set_xlabel("")
    perfil = d.groupby("hora_dia")[[OBJETIVO, "pred"]].mean()
    perfil.plot(ax=axes[1], marker="o")
    axes[1].set_title(f"{conjunto}: promedio por estación según la hora del día")
    axes[1].legend(["real", "predicho"])
    fig.tight_layout()
    ruta_real = carpeta / "p1_real_vs_predicho.png"
    fig.savefig(ruta_real, dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    d["residuo"].clip(-10, 10).plot.hist(bins=41, ax=axes[0])
    axes[0].set_title(f"{conjunto}: residuales (real − predicho, recortados a ±10)")
    d["residuo"].abs().groupby(d["hora_dia"]).mean().plot.bar(ax=axes[1])
    axes[1].set_title(f"{conjunto}: MAE por hora del día")
    axes[1].set_xlabel("hora")
    fig.tight_layout()
    ruta_res = carpeta / "p1_residuales.png"
    fig.savefig(ruta_res, dpi=110)
    plt.close(fig)
    return [ruta_real, ruta_res]


def entrenar_y_registrar(
    tabla, nombre, params, run_name, etapa, entrenar_con, nested=False, tags=None
) -> str:
    """Entrena una configuración, la evalúa y registra métricas, gráficas, config, datos y modelo."""
    with iniciar_run(
        "P1",
        run_name,
        model_family=FAMILIAS[nombre],
        stage_of_work=etapa,
        params={"modelo": nombre, "entrenado_con": "+".join(entrenar_con), **params},
        nested=nested,
        tags=tags,
    ) as run:
        evaluar_en = ["test"] if "val" in entrenar_con else ["val", "test"]
        entreno = tabla[tabla["conjunto"].isin(entrenar_con)]

        if nombre == "baseline":
            modelo, predecir = None, lambda d: d[PRED]
        else:
            modelo = crear_modelo(nombre, params)
            logger.info(f"P1: entrenando {run_name} con {len(entreno):,} filas")
            inicio = time.perf_counter()
            modelo.fit(entreno[CARACTERISTICAS], entreno[OBJETIVO])
            mlflow.log_metric("train_segundos", time.perf_counter() - inicio)
            predecir = lambda d: modelo.predict(d[CARACTERISTICAS])

        metricas, predicciones = {}, {}
        for conjunto in evaluar_en:
            d = tabla[tabla["conjunto"] == conjunto]
            predicciones[conjunto] = predecir(d)
            metricas.update(calcular_metricas(d[OBJETIVO], predicciones[conjunto], conjunto))
        logger.info(f"P1 {run_name}: {metricas}")

        config = {
            "pregunta": "P1",
            "modelo": nombre,
            "hiperparametros": params,
            "caracteristicas": CARACTERISTICAS if modelo is not None else [PRED],
            "objetivo": "salidas de la estación en la siguiente hora",
            "entrenado_con": entrenar_con,
            "evaluado_en": evaluar_en,
            "criterio_seleccion": "val_mae",
        }
        with tempfile.TemporaryDirectory() as tmp:
            graficas = graficar_resultados(
                tabla[tabla["conjunto"] == evaluar_en[0]],
                predicciones[evaluar_en[0]],
                evaluar_en[0],
                Path(tmp),
            )
            registrar_resultados(
                metricas,
                modelo=modelo,
                X_ejemplo=entreno[CARACTERISTICAS].dropna() if modelo is not None else None,
                graficas=graficas,
                config=config,
                datos_entrenamiento=entreno if modelo is not None else None,
                nombre_datos=f"p1-estacion-hora-{'+'.join(entrenar_con)}",
                tipos_confiables=TIPOS_CONFIABLES.get(nombre),
            )
        return run.info.run_id, metricas


def buscar_hgb(tabla, params) -> str:
    """Búsqueda en malla de HGB: un run padre y un run hijo por combinación (nested runs)."""
    malla = {k: v for k, v in params.items() if isinstance(v, list)} or BUSQUEDA_DEFAULT
    fijos = {k: v for k, v in params.items() if not isinstance(v, list)}
    combinaciones = [dict(zip(malla, valores)) for valores in itertools.product(*malla.values())]

    with iniciar_run(
        "P1",
        "p1-hgb-search",
        model_family="hgb",
        stage_of_work="search",
        params={"modelo": "hgb-search", "malla": json.dumps(malla), **fijos},
    ) as padre:
        resultados = []
        for combinacion in combinaciones:
            nombre_hijo = "p1-hgb-" + "-".join(f"{k}{v}" for k, v in combinacion.items())
            run_id, metricas = entrenar_y_registrar(
                tabla,
                "hgb",
                {**fijos, **combinacion},
                nombre_hijo,
                "search",
                ["train"],
                nested=True,
            )
            resultados.append({"run_id": run_id, "params": combinacion, **metricas})

        mejor = min(resultados, key=lambda r: r["val_mae"])
        mlflow.set_tags({"mejor_hijo_run_id": mejor["run_id"], "criterio_seleccion": "val_mae"})
        mlflow.log_params({f"mejor_{k}": v for k, v in mejor["params"].items()})
        mlflow.log_dict(
            {"malla": malla, "fijos": fijos, "resultados": resultados},
            "config/resultados_busqueda.json",
        )
        logger.success(
            f"P1: mejor combinación por val_mae: {mejor['params']} ({mejor['val_mae']:.4f})"
        )
        return padre.info.run_id


def _valor(texto: str):
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return texto


def mejores_params_busqueda() -> tuple[dict, str]:
    """Hiperparámetros ganadores (por val_mae) de la última búsqueda p1-hgb-search terminada."""
    mlflow.set_tracking_uri(tracking.MLFLOW_TRACKING_URI)
    experimento = os.getenv("MLFLOW_EXPERIMENT_ID") or (
        mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT).experiment_id
    )
    busquedas = mlflow.search_runs(
        experiment_ids=[experimento],
        filter_string="tags.mlflow.runName = 'p1-hgb-search' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not busquedas:
        raise ValueError(
            "No hay una búsqueda p1-hgb-search terminada: correr model=hgb-search antes"
        )
    busqueda = busquedas[0]
    mejores = {
        k.removeprefix("mejor_"): _valor(v)
        for k, v in busqueda.data.params.items()
        if k.startswith("mejor_")
    }
    fijos = {
        k: _valor(v)
        for k, v in busqueda.data.params.items()
        if k in HGB_DEFAULTS and k not in mejores
    }
    return {**fijos, **mejores}, busqueda.info.run_id


def train(model: str, params: dict) -> str:
    """Punto de entrada `train` del MLflow Project para P1.

    model: baseline | ridge | hgb | hgb-squared | hgb-search | hgb-final.
    Los candidatos y la búsqueda entrenan con train y reportan val y test (se elige con val).
    hgb-final reentrena con train + val la mejor combinación de la última búsqueda (o los
    params que se le pasen) y se evalúa una sola vez en test.
    Ninguna corrida necesita params: en Windows, `mlflow run` rompe las comillas del JSON.
    """
    tabla = construir_tabla()
    if model == "baseline":
        return entrenar_y_registrar(tabla, "baseline", {}, RUN_NAME, "baseline", ["train"])[0]
    if model == "ridge":
        nombre = f"p1-ridge-a{params.get('alpha', 1.0)}"
        return entrenar_y_registrar(tabla, "ridge", params, nombre, "candidate", ["train"])[0]
    if model in ("hgb", "hgb-squared"):
        if model == "hgb-squared":
            params = {"loss": "squared_error", **params}
        nombre = f"p1-hgb-{params.get('loss', HGB_DEFAULTS['loss'])}"
        return entrenar_y_registrar(tabla, "hgb", params, nombre, "candidate", ["train"])[0]
    if model == "hgb-search":
        return buscar_hgb(tabla, params)
    if model == "hgb-final":
        tags = None
        if not params:
            params, origen = mejores_params_busqueda()
            tags = {"busqueda_origen_run_id": origen}
            logger.info(
                f"P1: hgb-final con la mejor combinación de la búsqueda {origen}: {params}"
            )
        return entrenar_y_registrar(
            tabla, "hgb", params, "p1-hgb-final", "candidate", ["train", "val"], tags=tags
        )[0]
    raise ValueError(f"Modelo desconocido para P1: {model}. Opciones: {MODELOS}")


if __name__ == "__main__":
    main()
