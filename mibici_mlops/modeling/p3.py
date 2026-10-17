"""P3 - Baseline de clustering de estaciones MiBici.

Agrupa estaciones segun sus patrones de uso mediante K-Means.

Uso desde la raíz del repositorio:
    python -m mibici_mlops.modeling.p3
"""

from loguru import logger
import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mibici_mlops.tracking import registrar_run

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


if __name__ == "__main__":
    main()