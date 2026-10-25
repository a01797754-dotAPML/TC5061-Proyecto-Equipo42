"""P2 - Clasificación del riesgo de desabasto y saturación.

Ejecuta el baseline estacional y el modelo Random Forest,
registrando los resultados en MLflow.

Uso:
    python -m mibici_mlops.modeling.p2
"""

from mibici_mlops.entrenar_p2 import main as entrenar_p2


def main() -> str:
    """Ejecuta los experimentos P2 y devuelve el run ID del Random Forest."""
    return entrenar_p2()

def preparar_experimento_p2():
    """Carga o prepara las particiones P2 con caché local."""
    from mibici_mlops.modeling.p2_cache import cargar_o_preparar_p2

    return cargar_o_preparar_p2()


def train(model: str, params: dict) -> str:
    """Entrena P2 y registra sus resultados con la interfaz del equipo."""
    import mlflow

    from mibici_mlops.modeling.p2_candidates import crear_candidato_p2
    from mibici_mlops.modeling.p2_plots import registrar_graficas_p2
    from mibici_mlops.modeling.p2_search import (
        ejecutar_busqueda_p2,
        evaluar_candidato_p2,
    )
    from mibici_mlops.tracking import iniciar_run, registrar_resultados

    if model == "baseline":
        etapa = "baseline"
    elif model == "rf-search":
        etapa = "search"
    else:
        etapa = "candidate"

    with iniciar_run(
        pregunta="P2",
        run_name=f"p2-{model}",
        model_family="random_forest",
        stage_of_work=etapa,
        params={"modelo": model, **params},
    ) as run:
        print(f"Preparando experimento P2: {model}", flush=True)

        X_train, y_train, X_val, y_val, _X_test, _y_test = (
            preparar_experimento_p2()
        )

        if model == "rf-search":
            mejor = ejecutar_busqueda_p2(
                X_train, y_train, X_val, y_val
            )
            candidato = mejor["modelo"]
            metricas = mejor["metricas"]

            mlflow.set_tag("best_child_run_id", mejor["run_id"])
            mlflow.set_tag("best_candidate", mejor["nombre"])
            mlflow.log_params(mejor["params"])
        elif model == "baseline":
            from mibici_mlops.modelo_p2 import crear_modelo_p2

            candidato = crear_modelo_p2()
            print("Entrenando baseline P2", flush=True)
            candidato.fit(X_train, y_train)
            metricas = evaluar_candidato_p2(
                candidato, X_val, y_val
            )
        else:
            candidato = crear_candidato_p2(model, params)
            print(f"Entrenando candidato: {model}", flush=True)
            candidato.fit(X_train, y_train)
            metricas = evaluar_candidato_p2(
                candidato, X_val, y_val
            )

        configuracion = {
            "pregunta": "P2",
            "modelo": model,
            "etapa": etapa,
            "parametros_solicitados": params,
            "criterio_seleccion": "val_macro_f1",
            "conjunto_evaluacion": "validacion",
            "semilla": 42,
        }

        if model == "rf-search":
            configuracion["mejor_candidato"] = mejor["nombre"]
            configuracion["mejores_parametros"] = mejor["params"]
            configuracion["best_child_run_id"] = mejor["run_id"]

        registrar_resultados(
            metricas=metricas,
            modelo=candidato,
            X_ejemplo=X_train,
            config=configuracion,
            datos_entrenamiento=X_train.assign(riesgo=y_train.to_numpy()),
            nombre_datos="p2-estacion-hora-train",
            tipos_confiables=["sklearn.tree._tree.Tree"],
        )

        registrar_graficas_p2(
            candidato,
            X_val,
            y_val,
            model,
        )

        print(f"Métricas de validación P2: {metricas}", flush=True)
        return run.info.run_id


if __name__ == "__main__":
    main()
