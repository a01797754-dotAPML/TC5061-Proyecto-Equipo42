"""Configuración de la búsqueda de hiperparámetros para P2."""

from itertools import product


ESPACIO_BUSQUEDA_P2 = {
    "n_estimators": [50, 100],
    "max_depth": [12, 16],
    "min_samples_leaf": [10, 20],
}


def generar_combinaciones_p2() -> list[dict]:
    """Genera combinaciones reproducibles de hiperparámetros."""
    nombres = list(ESPACIO_BUSQUEDA_P2)

    return [
        dict(zip(nombres, valores))
        for valores in product(
            *(ESPACIO_BUSQUEDA_P2[nombre] for nombre in nombres)
        )
    ]

def evaluar_candidato_p2(modelo, X_val, y_val) -> dict:
    """Evalúa un candidato P2 utilizando únicamente validación."""
    from mibici_mlops.entrenar_p2 import evaluar_predicciones

    predicciones = modelo.predict(X_val)

    return evaluar_predicciones(
        y_val,
        predicciones,
        conjunto="val",
    )

def seleccionar_mejor_candidato_p2(resultados: list[dict]) -> dict:
    """Selecciona el mejor candidato usando solo métricas de validación."""

    if not resultados:
        raise ValueError("No hay candidatos para comparar")

    return max(
        resultados,
        key=lambda resultado: (
            resultado["metricas"]["val_macro_f1"],
            resultado["metricas"]["val_recall_desabasto"],
            resultado["metricas"]["val_recall_saturacion"],
        ),
    )

def ejecutar_busqueda_p2(X_train, y_train, X_val, y_val) -> dict:
    """Evalúa ocho combinaciones P2 mediante runs anidados de MLflow."""
    import mlflow

    from mibici_mlops.modeling.p2_candidates import crear_candidato_p2
    from mibici_mlops.tracking import iniciar_run, registrar_resultados

    if mlflow.active_run() is None:
        raise RuntimeError(
            "La búsqueda P2 requiere un run padre activo de MLflow"
        )

    resultados = []

    experimento_padre = mlflow.active_run().info.experiment_id
    mlflow.set_experiment(
        experiment_id=experimento_padre
    )

    for numero, parametros in enumerate(
        generar_combinaciones_p2(), start=1
    ):
        nombre = f"p2-rf-search-{numero:02d}"
        print(f"Entrenando {nombre}: {parametros}", flush=True)

        with iniciar_run(
            pregunta="P2",
            run_name=nombre,
            model_family="random_forest",
            stage_of_work="search",
            params=parametros,
            nested=True,
        ) as run:
            modelo = crear_candidato_p2("rf-ajustado-1", parametros)
            modelo.fit(X_train, y_train)

            metricas = evaluar_candidato_p2(modelo, X_val, y_val)

            registrar_resultados(
                metricas=metricas,
                registrar_manifiesto=False,
            )

            resultados.append({
                "run_id": run.info.run_id,
                "nombre": nombre,
                "params": parametros.copy(),
                "metricas": metricas,
            })

    mejor = seleccionar_mejor_candidato_p2(resultados)

    print(f"Mejor candidato P2: {mejor['nombre']}", flush=True)
    print(f"Métricas: {mejor['metricas']}", flush=True)

    mejor["modelo"] = crear_candidato_p2(
        "rf-ajustado-1",
        mejor["params"],
    )
    mejor["modelo"].fit(X_train, y_train)

    return mejor
