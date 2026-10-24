"""P3 - Baseline de clustering de estaciones MiBici.

Agrupa estaciones segun sus patrones de uso mediante K-Means.

Uso desde la raíz del repositorio:
    python -m mibici_mlops.modeling.p3
"""
from time import perf_counter

from loguru import logger
import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mibici_mlops.modeling.p3_candidates import crear_candidato_p3
from mibici_mlops.modeling.p3_plots import generar_graficas_p3
from mibici_mlops.modeling.p3_search import (
    ejecutar_busqueda_p3,
    evaluar_candidato_p3,
    transformar_para_metricas,
)
from mibici_mlops.tracking import (
    iniciar_run,
    registrar_resultados,
    registrar_run,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mibici_mlops.config import ESTACIONES_PATH, FIGURES_DIR, SEED, VIAJES_PATH
from mibici_mlops.features import (
    agregar_capacidad,
    agregar_estacion_hora,
    crear_perfil_p3,
    particion_temporal,
)

RUN_NAME = "p3-baseline-kmeans"
K_CANDIDATOS = range(3, 9)
VARIANZA_PCA = 0.90

def cargar_datos_p3() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los viajes procesados y la nomenclatura de estaciones."""
    viajes = pd.read_parquet(
        VIAJES_PATH,
        columns=[
            "inicio_viaje",
            "fin_viaje",
            "origen_id",
            "destino_id",
        ],
    )

    estaciones = pd.read_parquet(ESTACIONES_PATH)

    return viajes, estaciones


def construir_perfiles_p3(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Construye perfiles independientes para train, validacion y test.

    Cada periodo se transforma por separado para evitar incorporar
    información futura en los perfiles utilizados durante entrenamiento.
    """
    logger.info("P3: construyendo tabla estación-hora")

    tabla = agregar_estacion_hora(viajes)
    tabla = agregar_capacidad(tabla, estaciones)
    tabla = particion_temporal(tabla)

    perfiles = {}

    for conjunto in ["train", "val", "test"]:
        logger.info(f"P3: construyendo perfil {conjunto}")

        datos = tabla[tabla["conjunto"] == conjunto].copy()

        perfiles[conjunto] = crear_perfil_p3(datos)

    return perfiles

def escalar_perfiles_p3(
    perfiles: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    """Escala las features de P3 usando únicamente el periodo de entrenamiento.

    El StandardScaler se ajusta con train y luego se reutiliza sin reentrenar
    para transformar validation y test. Esto evita fuga de información futura.

    `estacion_id` se conserva como identificador, pero no entra al escalado ni
    posteriormente al clustering.
    """
    columnas_features = [
        columna
        for columna in perfiles["train"].columns
        if columna != "estacion_id"
    ]

    for conjunto in ["val", "test"]:
        columnas_conjunto = [
            columna
            for columna in perfiles[conjunto].columns
            if columna != "estacion_id"
        ]

        if columnas_conjunto != columnas_features:
            raise ValueError(
                f"Las features de {conjunto} no coinciden con las de train"
            )

    scaler = StandardScaler()

    escalados = {}

    for conjunto in ["train", "val", "test"]:
        perfil = perfiles[conjunto]

        if conjunto == "train":
            valores = scaler.fit_transform(perfil[columnas_features])
        else:
            valores = scaler.transform(perfil[columnas_features])

        escalado = pd.DataFrame(
            valores,
            columns=columnas_features,
            index=perfil.index,
        )

        escalado.insert(
            0,
            "estacion_id",
            perfil["estacion_id"].to_numpy(),
        )

        escalados[conjunto] = escalado

    return escalados, scaler

def reducir_dimensiones_pca(
    escalados: dict[str, pd.DataFrame],
    varianza_objetivo: float = VARIANZA_PCA,
) -> tuple[dict[str, pd.DataFrame], PCA]:
    """Reduce las features de P3 con PCA ajustado únicamente sobre train.

    Se conservan suficientes componentes para explicar al menos el porcentaje
    de varianza indicado. Validation y test se transforman usando el mismo PCA
    aprendido con train, evitando fuga de información futura.
    """
    columnas_features = [
        columna
        for columna in escalados["train"].columns
        if columna != "estacion_id"
    ]

    pca = PCA(
        n_components=varianza_objetivo,
        svd_solver="full",
    )

    X_train_pca = pca.fit_transform(
        escalados["train"][columnas_features]
    )

    nombres_componentes = [
        f"pc_{i + 1:02d}"
        for i in range(X_train_pca.shape[1])
    ]

    reducidos = {}

    for conjunto in ["train", "val", "test"]:
        perfil = escalados[conjunto]

        if conjunto == "train":
            valores = X_train_pca
        else:
            valores = pca.transform(
                perfil[columnas_features]
            )

        reducido = pd.DataFrame(
            valores,
            columns=nombres_componentes,
            index=perfil.index,
        )

        reducido.insert(
            0,
            "estacion_id",
            perfil["estacion_id"].to_numpy(),
        )

        reducidos[conjunto] = reducido

    logger.info(
        "P3 PCA: "
        f"{len(columnas_features)} features -> "
        f"{len(nombres_componentes)} componentes; "
        f"varianza explicada="
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    return reducidos, pca

def evaluar_kmeans_p3(
    escalados: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, int]:
    """Evalúa K-Means para k=3...8 y selecciona k con validation silhouette.

    Cada modelo se ajusta exclusivamente con el perfil de entrenamiento.
    Validation se asigna a los centroides aprendidos con train y se utiliza
    para seleccionar el número de clusters sin consultar el conjunto de test.

    Devuelve una tabla con métricas por k y el mejor valor de k.
    """
    columnas_features = [
        columna
        for columna in escalados["train"].columns
        if columna != "estacion_id"
    ]

    X_train = escalados["train"][columnas_features]
    X_val = escalados["val"][columnas_features]

    resultados = []

    for k in K_CANDIDATOS:
        logger.info(f"P3: evaluando KMeans con k={k}")

        modelo = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=20,
        )

        etiquetas_train = modelo.fit_predict(X_train)
        etiquetas_val = modelo.predict(X_val)

        train_silhouette = silhouette_score(
            X_train,
            etiquetas_train,
        )

        train_davies_bouldin = davies_bouldin_score(
            X_train,
            etiquetas_train,
        )

        if len(set(etiquetas_val)) > 1:
            val_silhouette = silhouette_score(
                X_val,
                etiquetas_val,
            )

            val_davies_bouldin = davies_bouldin_score(
                X_val,
                etiquetas_val,
            )
        else:
            val_silhouette = float("nan")
            val_davies_bouldin = float("nan")

        resultados.append(
            {
                "k": k,
                "train_silhouette": train_silhouette,
                "train_davies_bouldin": train_davies_bouldin,
                "val_silhouette": val_silhouette,
                "val_davies_bouldin": val_davies_bouldin,
            }
        )

    resultados = pd.DataFrame(resultados)

    mejor_fila = resultados["val_silhouette"].idxmax()
    mejor_k = int(resultados.loc[mejor_fila, "k"])

    logger.info(f"P3: mejor k según validation silhouette = {mejor_k}")

    return resultados, mejor_k

def construir_perfil_final_p3(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el perfil final de P3 con train + validation.

    Después de seleccionar k utilizando validation, el modelo final se
    reentrena utilizando los primeros 12 meses del proyecto:
    julio de 2025 a junio de 2026.

    El perfil se recalcula directamente sobre ese periodo completo en lugar
    de promediar perfiles previamente agregados, para conservar correctamente
    la distribución real de salidas y llegadas por hora.
    """
    tabla = agregar_estacion_hora(viajes)
    tabla = agregar_capacidad(tabla, estaciones)
    tabla = particion_temporal(tabla)

    datos_12m = tabla[
        tabla["conjunto"].isin(["train", "val"])
    ].copy()

    return crear_perfil_p3(datos_12m)

def entrenar_modelo_final_p3(
    perfil_final: pd.DataFrame,
    perfil_test: pd.DataFrame,
    k: int,
) -> tuple[KMeans, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """Entrena el K-Means final con train + validation y transforma test.

    El scaler se ajusta únicamente con el perfil final de los primeros
    12 meses. Después se reutiliza para transformar el perfil de test.

    `estacion_id` se conserva solo como identificador y no entra al modelo.
    """
    columnas_features = [
        columna
        for columna in perfil_final.columns
        if columna != "estacion_id"
    ]

    columnas_test = [
        columna
        for columna in perfil_test.columns
        if columna != "estacion_id"
    ]

    if columnas_test != columnas_features:
        raise ValueError(
            "Las features de test no coinciden con las del entrenamiento final"
        )

    scaler = StandardScaler()

    X_final = scaler.fit_transform(
        perfil_final[columnas_features]
    )

    X_test = scaler.transform(
        perfil_test[columnas_features]
    )

    modelo = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=20,
    )

    labels_final = modelo.fit_predict(X_final)
    labels_test = modelo.predict(X_test)

    resultado_final = perfil_final[["estacion_id"]].copy()
    resultado_final["cluster"] = labels_final

    resultado_test = perfil_test[["estacion_id"]].copy()
    resultado_test["cluster"] = labels_test

    return modelo, scaler, resultado_final, resultado_test

def entrenar_modelo_final_p3_pca(
    perfil_final: pd.DataFrame,
    perfil_test: pd.DataFrame,
    k: int,
    varianza_objetivo: float = VARIANZA_PCA,
) -> tuple[
    KMeans,
    StandardScaler,
    PCA,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Entrena el candidato PCA + KMeans con train + validation.

    El scaler y PCA se ajustan únicamente sobre el perfil final de 12 meses.
    Test se transforma usando los objetos ya ajustados, sin fuga de información.
    """
    columnas_features = [
        columna
        for columna in perfil_final.columns
        if columna != "estacion_id"
    ]

    columnas_test = [
        columna
        for columna in perfil_test.columns
        if columna != "estacion_id"
    ]

    if columnas_test != columnas_features:
        raise ValueError(
            "Las features de test no coinciden con las del entrenamiento final"
        )

    scaler = StandardScaler()

    X_final_escalado = scaler.fit_transform(
        perfil_final[columnas_features]
    )

    X_test_escalado = scaler.transform(
        perfil_test[columnas_features]
    )

    pca = PCA(
        n_components=varianza_objetivo,
        svd_solver="full",
    )

    X_final_pca = pca.fit_transform(
        X_final_escalado
    )

    X_test_pca = pca.transform(
        X_test_escalado
    )

    modelo = KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=20,
    )

    labels_final = modelo.fit_predict(X_final_pca)
    labels_test = modelo.predict(X_test_pca)

    resultado_final = perfil_final[["estacion_id"]].copy()
    resultado_final["cluster"] = labels_final

    resultado_test = perfil_test[["estacion_id"]].copy()
    resultado_test["cluster"] = labels_test

    logger.info(
        "P3 PCA final: "
        f"{len(columnas_features)} features -> "
        f"{pca.n_components_} componentes; "
        f"varianza explicada="
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    return (
        modelo,
        scaler,
        pca,
        resultado_final,
        resultado_test,
    )

def evaluar_modelo_final_p3(
    modelo: KMeans,
    scaler: StandardScaler,
    perfil_final: pd.DataFrame,
    perfil_test: pd.DataFrame,
    resultado_final: pd.DataFrame,
    resultado_test: pd.DataFrame,
) -> dict:
    """Evalúa separación y estabilidad temporal del baseline final de P3.

    Silhouette y Davies-Bouldin se calculan sobre el perfil utilizado para
    entrenar el modelo final (train + validation).

    La estabilidad en test se mide con Adjusted Rand Index únicamente sobre
    estaciones presentes tanto en el periodo final de entrenamiento como en
    test.
    """
    columnas_features = [
        columna
        for columna in perfil_final.columns
        if columna != "estacion_id"
    ]

    X_final = scaler.transform(
        perfil_final[columnas_features]
    )

    labels_final = modelo.predict(X_final)

    train_silhouette = silhouette_score(
        X_final,
        labels_final,
    )

    train_davies_bouldin = davies_bouldin_score(
        X_final,
        labels_final,
    )

    comparables = resultado_final.merge(
        resultado_test,
        on="estacion_id",
        suffixes=("_final", "_test"),
    )

    test_estabilidad_ari = adjusted_rand_score(
        comparables["cluster_final"],
        comparables["cluster_test"],
    )

    return {
        "train_silhouette": train_silhouette,
        "train_davies_bouldin": train_davies_bouldin,
        "test_estabilidad_ari": test_estabilidad_ari,
        "test_estaciones_comparables": float(len(comparables)),
    }

def evaluar_modelo_final_p3_pca(
    modelo: KMeans,
    scaler: StandardScaler,
    pca: PCA,
    perfil_final: pd.DataFrame,
    resultado_final: pd.DataFrame,
    resultado_test: pd.DataFrame,
) -> dict:
    """Evalúa separación y estabilidad temporal de PCA + KMeans."""

    columnas_features = [
        columna
        for columna in perfil_final.columns
        if columna != "estacion_id"
    ]

    X_final_escalado = scaler.transform(
        perfil_final[columnas_features]
    )

    X_final_pca = pca.transform(
        X_final_escalado
    )

    labels_final = modelo.predict(
        X_final_pca
    )

    train_silhouette = silhouette_score(
        X_final_pca,
        labels_final,
    )

    train_davies_bouldin = davies_bouldin_score(
        X_final_pca,
        labels_final,
    )

    comparables = resultado_final.merge(
        resultado_test,
        on="estacion_id",
        suffixes=("_final", "_test"),
    )

    test_estabilidad_ari = adjusted_rand_score(
        comparables["cluster_final"],
        comparables["cluster_test"],
    )

    return {
        "train_silhouette": train_silhouette,
        "train_davies_bouldin": train_davies_bouldin,
        "test_estabilidad_ari": test_estabilidad_ari,
        "test_estaciones_comparables": float(len(comparables)),
    }

def graficar_seleccion_k(
    resultados: pd.DataFrame,
    ruta,
) -> None:
    """Grafica las métricas usadas para comparar valores de k en P3."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        resultados["k"],
        resultados["train_silhouette"],
        marker="o",
        label="train",
    )
    axes[0].plot(
        resultados["k"],
        resultados["val_silhouette"],
        marker="o",
        label="validation",
    )
    axes[0].set_title("P3: Silhouette por número de clusters")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette")
    axes[0].legend()

    axes[1].plot(
        resultados["k"],
        resultados["train_davies_bouldin"],
        marker="o",
        label="train",
    )
    axes[1].plot(
        resultados["k"],
        resultados["val_davies_bouldin"],
        marker="o",
        label="validation",
    )
    axes[1].set_title("P3: Davies-Bouldin por número de clusters")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Davies-Bouldin")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_mapa_clusters(
    resultado: pd.DataFrame,
    estaciones: pd.DataFrame,
    ruta,
) -> None:
    """Visualiza espacialmente las estaciones según el cluster asignado.

    Latitud y longitud se utilizan únicamente para interpretar los grupos
    después del clustering; no forman parte de las features del modelo.
    """
    mapa = resultado.merge(
        estaciones[
            [
                "estacion_id",
                "nombre",
                "latitud",
                "longitud",
            ]
        ],
        on="estacion_id",
        how="left",
    )

    if mapa[["latitud", "longitud"]].isna().any().any():
        raise ValueError(
            "Hay estaciones sin coordenadas para generar el mapa de P3"
        )

    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(
        mapa["longitud"],
        mapa["latitud"],
        c=mapa["cluster"],
        cmap="tab10",
        s=30,
        alpha=0.8,
    )

    ax.set_title("P3: distribución geográfica de clusters de estaciones")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    leyenda = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
    )
    ax.add_artist(leyenda)

    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
def preparar_experimento_p3():
    """Prepara perfiles y matrices de P3 para la Entrega 2."""
    viajes, estaciones = cargar_datos_p3()

    perfiles = construir_perfiles_p3(
        viajes,
        estaciones,
    )

    columnas = [
        columna
        for columna in perfiles["train"].columns
        if columna != "estacion_id"
    ]

    X_train = perfiles["train"][columnas].copy()
    X_val = perfiles["val"][columnas].copy()
    X_test = perfiles["test"][columnas].copy()

    return {
        "viajes": viajes,
        "estaciones": estaciones,
        "perfiles": perfiles,
        "columnas": columnas,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
    }


def construir_resultado_clusters_p3(
    perfil: pd.DataFrame,
    etiquetas,
) -> pd.DataFrame:
    """Asocia cada estación con el cluster predicho."""
    resultado = perfil[["estacion_id"]].copy()
    resultado["cluster"] = etiquetas

    return resultado


def evaluar_pipeline_final_p3(
    modelo,
    perfil_final: pd.DataFrame,
    perfil_test: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evalúa separación interna y estabilidad temporal de un pipeline P3."""
    columnas = [
        columna
        for columna in perfil_final.columns
        if columna != "estacion_id"
    ]

    X_final = perfil_final[columnas]
    X_test = perfil_test[columnas]

    etiquetas_final = modelo.predict(X_final)
    etiquetas_test = modelo.predict(X_test)

    X_final_transformado = transformar_para_metricas(
        modelo,
        X_final,
    )

    metricas = {
        "train_silhouette": float(
            silhouette_score(
                X_final_transformado,
                etiquetas_final,
            )
        ),
        "train_davies_bouldin": float(
            davies_bouldin_score(
                X_final_transformado,
                etiquetas_final,
            )
        ),
    }

    resultado_final = construir_resultado_clusters_p3(
        perfil_final,
        etiquetas_final,
    )

    resultado_test = construir_resultado_clusters_p3(
        perfil_test,
        etiquetas_test,
    )

    comparables = resultado_final.merge(
        resultado_test,
        on="estacion_id",
        suffixes=("_final", "_test"),
    )

    metricas["test_estabilidad_ari"] = float(
        adjusted_rand_score(
            comparables["cluster_final"],
            comparables["cluster_test"],
        )
    )

    metricas["test_estaciones_comparables"] = float(
        len(comparables)
    )

    return (
        metricas,
        resultado_final,
        resultado_test,
    )
def train(model: str, params: dict) -> str:
    """Entrena y registra un experimento P3 mediante el MLflow Project.

    Modelos disponibles:
    - baseline
    - kmeans-sin-escalar
    - kmeans-flujo-escalado
    - pca-flujo-escalado
    - kmeans-search
    """
    modelos_validos = {
        "baseline",
        "kmeans-sin-escalar",
        "kmeans-flujo-escalado",
        "pca-flujo-escalado",
        "kmeans-search",
    }

    if model not in modelos_validos:
        raise ValueError(
            f"Modelo P3 desconocido: {model}. "
            f"Opciones: {sorted(modelos_validos)}"
        )

    datos = preparar_experimento_p3()

    viajes = datos["viajes"]
    estaciones = datos["estaciones"]
    perfiles = datos["perfiles"]
    columnas = datos["columnas"]

    X_train = datos["X_train"]
    X_val = datos["X_val"]

    if model == "baseline":
        etapa = "baseline"
        model_family = "kmeans"
    elif model == "kmeans-search":
        etapa = "search"
        model_family = "kmeans"
    elif model == "pca-flujo-escalado":
        etapa = "candidate"
        model_family = "pca_kmeans"
    else:
        etapa = "candidate"
        model_family = "kmeans"

    run_name = f"p3-{model}"

    params_run = {
        "modelo": model,
        **params,
    }

    with iniciar_run(
        pregunta="P3",
        run_name=run_name,
        model_family=model_family,
        stage_of_work=etapa,
        params=params_run,
    ) as run:
        if model == "kmeans-search":
            mejor = ejecutar_busqueda_p3(
                X_train=X_train,
                X_val=X_val,
                columnas=columnas,
            )

            parametros_modelo = mejor["params"]

            metricas_val = mejor["metricas"]

            import mlflow

            mlflow.set_tag(
                "best_child_run_id",
                mejor["run_id"],
            )
            mlflow.set_tag(
                "best_candidate",
                mejor["nombre"],
            )
            mlflow.log_params(
                {
                    "best_k": parametros_modelo["k"],
                }
            )

            nombre_modelo_final = "kmeans-flujo-escalado"

        else:
            parametros_modelo = params.copy()

            modelo_seleccion = crear_candidato_p3(
                model,
                columnas=columnas,
                params=parametros_modelo,
            )

            inicio_entrenamiento = perf_counter()

            modelo_seleccion.fit(X_train)

            tiempo_entrenamiento = (
                perf_counter()
                - inicio_entrenamiento
            )

            metricas_val = evaluar_candidato_p3(
                modelo_seleccion,
                X_val,
            )

            metricas_val["train_seconds_selection"] = float(
                tiempo_entrenamiento
            )

            nombre_modelo_final = model

        perfil_final = construir_perfil_final_p3(
            viajes,
            estaciones,
        )

        columnas_final = [
            columna
            for columna in perfil_final.columns
            if columna != "estacion_id"
        ]

        X_final = perfil_final[columnas_final]

        modelo_final = crear_candidato_p3(
            nombre_modelo_final,
            columnas=columnas_final,
            params=parametros_modelo,
        )

        inicio_final = perf_counter()

        modelo_final.fit(X_final)

        tiempo_final = perf_counter() - inicio_final

        (
            metricas_finales,
            resultado_final,
            _resultado_test,
        ) = evaluar_pipeline_final_p3(
            modelo_final,
            perfil_final,
            perfiles["test"],
        )

        inicio_inferencia = perf_counter()

        modelo_final.predict(
            perfiles["test"][columnas_final]
        )

        tiempo_inferencia = (
            perf_counter()
            - inicio_inferencia
        )

        n_test = len(perfiles["test"])

        metricas = {
            **metricas_val,
            **metricas_finales,
            "train_seconds_final": float(
                tiempo_final
            ),
            "test_inference_ms_per_station": float(
                (
                    tiempo_inferencia
                    / max(n_test, 1)
                )
                * 1000
            ),
        }

        graficas = generar_graficas_p3(
            perfil=perfil_final,
            resultado=resultado_final,
            estaciones=estaciones,
            nombre_modelo=model,
        )

        configuracion = {
            "pregunta": "P3",
            "modelo": model,
            "model_family": model_family,
            "stage_of_work": etapa,
            "parametros_solicitados": params,
            "parametros_modelo_final": parametros_modelo,
            "criterio_seleccion": "val_silhouette",
            "criterio_desempate": "val_davies_bouldin",
            "test_usado_para_seleccion": False,
            "n_features": len(columnas_final),
            "n_estaciones_train": len(
                perfiles["train"]
            ),
            "n_estaciones_val": len(
                perfiles["val"]
            ),
            "n_estaciones_final": len(
                perfil_final
            ),
            "n_estaciones_test": len(
                perfiles["test"]
            ),
            "ubicacion_en_clustering": False,
        }

        if model == "kmeans-search":
            configuracion.update(
                {
                    "best_child_run_id": mejor[
                        "run_id"
                    ],
                    "mejor_k": parametros_modelo[
                        "k"
                    ],
                }
            )

        registrar_resultados(
            metricas=metricas,
            modelo=modelo_final,
            X_ejemplo=X_final,
            graficas=graficas,
            config=configuracion,
            datos_entrenamiento=perfil_final,
            nombre_datos="p3-perfil-estacion-train-val",
        )

        logger.info(
            f"P3 {model} métricas: {metricas}"
        )

        return run.info.run_id
def main(run_name: str = RUN_NAME) -> str:
    """Ejecuta y registra en MLflow el baseline completo de P3."""

    logger.info("P3: cargando datos")
    viajes, estaciones = cargar_datos_p3()

    # 1. Construir perfiles temporales.
    perfiles = construir_perfiles_p3(viajes, estaciones)

    # 2. Escalar usando únicamente train y seleccionar k con validation.
    escalados, _ = escalar_perfiles_p3(perfiles)
    resultados_k, mejor_k = evaluar_kmeans_p3(escalados)

    # 3. Artefacto de selección de k.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ruta_seleccion_k = FIGURES_DIR / "p3_seleccion_k.png"
    graficar_seleccion_k(
        resultados_k,
        ruta_seleccion_k,
    )

    # 4. Reconstruir el perfil final con train + validation.
    perfil_final = construir_perfil_final_p3(
        viajes,
        estaciones,
    )

    # 5. Entrenar modelo final y asignar clusters a test.
    modelo, scaler, resultado_final, resultado_test = (
        entrenar_modelo_final_p3(
            perfil_final,
            perfiles["test"],
            mejor_k,
        )
    )

    # 6. Evaluación final.
    metricas = evaluar_modelo_final_p3(
        modelo,
        scaler,
        perfil_final,
        perfiles["test"],
        resultado_final,
        resultado_test,
    )

    logger.info(f"P3 métricas finales: {metricas}")

    # 7. Mapa de clusters.
    ruta_mapa = FIGURES_DIR / "p3_mapa_clusters.png"

    graficar_mapa_clusters(
        resultado_final,
        estaciones,
        ruta_mapa,
    )

    # 8. Parámetros reproducibles específicos de P3.
    params = {
        "modelo": "KMeans",
        "k": mejor_k,
        "k_candidatos": "3,4,5,6,7,8",
        "criterio_seleccion_k": "val_silhouette",
        "n_init": 20,
        "escalador": "StandardScaler",
        "n_features": len(perfil_final.columns) - 1,
        "n_estaciones_train": len(perfiles["train"]),
        "n_estaciones_val": len(perfiles["val"]),
        "n_estaciones_final_12m": len(perfil_final),
        "n_estaciones_test": len(perfiles["test"]),
        "granularidad": "estacion",
        "ubicacion_en_clustering": False,
    }

    # 9. Registrar baseline en MLflow.
    return registrar_run(
        pregunta="P3",
        run_name=run_name,
        params=params,
        metricas=metricas,
        artefactos=[
            ruta_seleccion_k,
            ruta_mapa,
        ],
        modelo=modelo,
    )

def main_pca(run_name: str = "p3-pca-kmeans") -> str:
    """Ejecuta y registra en MLflow el candidato PCA + KMeans de P3."""

    logger.info("P3 PCA: cargando datos")
    viajes, estaciones = cargar_datos_p3()

    # 1. Construir perfiles train / val / test.
    perfiles = construir_perfiles_p3(
        viajes,
        estaciones,
    )

    # 2. Escalar con train y ajustar PCA únicamente con train.
    escalados, _ = escalar_perfiles_p3(perfiles)

    reducidos, pca_seleccion = reducir_dimensiones_pca(
        escalados
    )

    # 3. Seleccionar k usando validation.
    _resultados_k, mejor_k = evaluar_kmeans_p3(
        reducidos
    )
    # 4. Construir perfil final train + validation.
    perfil_final = construir_perfil_final_p3(
        viajes,
        estaciones,
    )

    # 5. Entrenar scaler + PCA + KMeans final.
    (
        modelo,
        scaler,
        pca,
        resultado_final,
        resultado_test,
    ) = entrenar_modelo_final_p3_pca(
        perfil_final,
        perfiles["test"],
        mejor_k,
    )

    # 6. Evaluación final.
    metricas = evaluar_modelo_final_p3_pca(
        modelo,
        scaler,
        pca,
        perfil_final,
        resultado_final,
        resultado_test,
    )

    logger.info(
        f"P3 PCA métricas finales: {metricas}"
    )

    # 7. Artefacto geográfico.
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_mapa = (
        FIGURES_DIR
        / "p3_pca_mapa_clusters.png"
    )

    graficar_mapa_clusters(
        resultado_final,
        estaciones,
        ruta_mapa,
    )

    # 8. Parámetros reproducibles.
    params = {
        "modelo": "PCA+KMeans",
        "k": mejor_k,
        "k_candidatos": "3,4,5,6,7,8",
        "criterio_seleccion_k": "val_silhouette",
        "n_init": 20,
        "escalador": "StandardScaler",
        "pca_varianza_objetivo": VARIANZA_PCA,
        "pca_componentes_train": pca_seleccion.n_components_,
        "pca_componentes_final": pca.n_components_,
        "pca_varianza_final": float(
            pca.explained_variance_ratio_.sum()
        ),
        "n_features_originales": len(perfil_final.columns) - 1,
        "n_estaciones_train": len(perfiles["train"]),
        "n_estaciones_val": len(perfiles["val"]),
        "n_estaciones_final_12m": len(perfil_final),
        "n_estaciones_test": len(perfiles["test"]),
        "granularidad": "estacion",
        "ubicacion_en_clustering": False,
    }

    return registrar_run(
        pregunta="P3",
        run_name=run_name,
        params=params,
        metricas=metricas,
        artefactos=[
            ruta_mapa,
        ],
        modelo=modelo,
    )

if __name__ == "__main__":
    main()