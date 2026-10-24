"""Búsqueda de hiperparámetros para P3 mediante nested runs de MLflow."""

import math

import mlflow
from sklearn.metrics import davies_bouldin_score, silhouette_score

from mibici_mlops.modeling.p3_candidates import crear_candidato_p3
from mibici_mlops.tracking import iniciar_run, registrar_resultados

K_BUSQUEDA = (3, 4, 5, 6, 7, 8)


def transformar_para_metricas(modelo, X):
    """Transforma X al espacio donde KMeans realiza el clustering."""
    if len(modelo.steps) == 1:
        return X

    return modelo[:-1].transform(X)


def evaluar_candidato_p3(modelo, X_val) -> dict:
    """Evalúa un modelo P3 usando únicamente el conjunto de validación."""
    etiquetas = modelo.predict(X_val)

    if len(set(etiquetas)) < 2:
        return {
            "val_silhouette": float("nan"),
            "val_davies_bouldin": float("nan"),
        }

    X_transformado = transformar_para_metricas(modelo, X_val)

    return {
        "val_silhouette": float(
            silhouette_score(
                X_transformado,
                etiquetas,
            )
        ),
        "val_davies_bouldin": float(
            davies_bouldin_score(
                X_transformado,
                etiquetas,
            )
        ),
    }


def seleccionar_mejor_candidato_p3(resultados: list[dict]) -> dict:
    """Selecciona el mejor candidato usando solo validation.

    Prioridad:
    1. mayor val_silhouette;
    2. menor val_davies_bouldin en caso de empate.
    """
    validos = [
        resultado
        for resultado in resultados
        if not math.isnan(
            resultado["metricas"]["val_silhouette"]
        )
    ]

    if not validos:
        raise ValueError(
            "No hay resultados válidos de P3 para seleccionar"
        )

    return max(
        validos,
        key=lambda resultado: (
            resultado["metricas"]["val_silhouette"],
            -resultado["metricas"]["val_davies_bouldin"],
        ),
    )


def ejecutar_busqueda_p3(
    X_train,
    X_val,
    columnas: list[str],
    modelo_base: str = "kmeans-flujo-escalado",
) -> dict:
    """Busca el mejor k mediante runs anidados de MLflow.

    El run padre debe estar abierto antes de llamar esta función.
    Cada valor de k genera un run hijo.
    La selección utiliza exclusivamente métricas de validación.
    """
    if mlflow.active_run() is None:
        raise RuntimeError(
            "La búsqueda P3 requiere un run padre activo de MLflow"
        )

    resultados = []

    for k in K_BUSQUEDA:
        nombre_run = f"p3-kmeans-search-k{k}"

        print(
            f"P3 búsqueda: entrenando {nombre_run}",
            flush=True,
        )

        with iniciar_run(
            pregunta="P3",
            run_name=nombre_run,
            model_family="kmeans",
            stage_of_work="search",
            params={
                "modelo": modelo_base,
                "k": k,
            },
            nested=True,
        ) as run:
            modelo = crear_candidato_p3(
                modelo_base,
                columnas=columnas,
                params={"k": k},
            )

            modelo.fit(X_train)

            metricas = evaluar_candidato_p3(
                modelo,
                X_val,
            )

            registrar_resultados(
                metricas=metricas,
                registrar_manifiesto=False,
            )

            resultados.append(
                {
                    "run_id": run.info.run_id,
                    "nombre": nombre_run,
                    "params": {"k": k},
                    "metricas": metricas,
                }
            )

    mejor = seleccionar_mejor_candidato_p3(
        resultados
    )

    mejor_modelo = crear_candidato_p3(
        modelo_base,
        columnas=columnas,
        params=mejor["params"],
    )
    mejor_modelo.fit(X_train)

    mejor["modelo"] = mejor_modelo

    print(
        "Mejor candidato de búsqueda P3: "
        f"{mejor['nombre']} · "
        f"silhouette={mejor['metricas']['val_silhouette']:.4f}",
        flush=True,
    )

    return mejor