"""Entrenamiento reproducible del modelo P2 de MiBici."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from mibici_mlops.modelo_p2 import (
    VARIABLES_P2,
    crear_modelo_p2,
    guardar_modelo_p2,
    preparar_datos_modelo,
)
from mibici_mlops.preparar_p2 import generar_dataset_p2
from mibici_mlops.tracking import registrar_run


def evaluar_predicciones(y_real, y_pred, conjunto):
    """Calcula las métricas del baseline estacional P2."""
    metricas = {
        f"{conjunto}_accuracy": float(accuracy_score(y_real, y_pred)),
        f"{conjunto}_macro_f1": float(
            f1_score(
                y_real,
                y_pred,
                average="macro",
                labels=[0, 1, 2],
                zero_division=0,
            )
        ),
    }

    for clase, nombre in [(1, "desabasto"), (2, "saturacion")]:
        metricas[f"{conjunto}_recall_{nombre}"] = float(
            recall_score(
                y_real,
                y_pred,
                labels=[clase],
                average="macro",
                zero_division=0,
            )
        )

        metricas[f"{conjunto}_precision_{nombre}"] = float(
            precision_score(
                y_real,
                y_pred,
                labels=[clase],
                average="macro",
                zero_division=0,
            )
        )

    return metricas


def evaluar_modelo(modelo, X_val, y_val, X_test, y_test):
    """Calcula métricas de clasificación en validación y test."""
    metricas = {}

    for conjunto, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        predicciones = modelo.predict(X)

        metricas[f"{conjunto}_accuracy"] = float(accuracy_score(y, predicciones))
        metricas[f"{conjunto}_macro_f1"] = float(
            f1_score(
                y,
                predicciones,
                average="macro",
                labels=[0, 1, 2],
                zero_division=0,
            )
        )
        for clase, nombre in [(1, "desabasto"), (2, "saturacion")]:
            metricas[f"{conjunto}_recall_{nombre}"] = float(
                recall_score(
                    y,
                    predicciones,
                    labels=[clase],
                    average="macro",
                    zero_division=0,
                )
            )

            metricas[f"{conjunto}_precision_{nombre}"] = float(
                precision_score(
                    y,
                    predicciones,
                    labels=[clase],
                    average="macro",
                    zero_division=0,
                )
            )

    return metricas


def guardar_matriz_confusion(modelo, X_test, y_test):
    """Genera la matriz de confusión de P2 sobre el conjunto de test."""
    from mibici_mlops.config import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ruta = FIGURES_DIR / "p2_matriz_confusion_test.png"

    predicciones = modelo.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predicciones,
        labels=[0, 1, 2],
        display_labels=["Bajo riesgo", "Desabasto", "Saturación"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    ax.set_title("P2 - Matriz de confusión (test)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

    return ruta


def main():
    print("Preparando dataset P2...", flush=True)
    tabla = generar_dataset_p2()

    X_train, y_train, X_val, y_val, X_test, y_test = preparar_datos_modelo(tabla)

    print(f"Registros de entrenamiento: {len(X_train):,}")
    print(f"Registros de validación: {len(X_val):,}")
    print(f"Registros de prueba reservados: {len(X_test):,}")

    # Baseline P2: clase observada en la misma estación
    # y hora de la semana anterior (168 horas).
    print("Evaluando baseline estacional P2...", flush=True)

    tabla_baseline = tabla.dropna(
        subset=VARIABLES_P2 + ["riesgo_p2_t_3h", "riesgo_p2_lag_165"]
    ).copy()

    metricas_baseline = {}

    for conjunto in ["val", "test"]:
        datos = tabla_baseline[tabla_baseline["conjunto"] == conjunto]

        metricas_baseline.update(
            evaluar_predicciones(
                datos["riesgo_p2_t_3h"].astype(int),
                datos["riesgo_p2_lag_165"].astype(int),
                conjunto,
            )
        )

    print(f"Metricas baseline P2: {metricas_baseline}")

    registrar_run(
        pregunta="P2",
        run_name="p2-baseline-semana-anterior",
        params={
            "modelo": "naive_estacional",
            "rezago_horas": 165,
            "objetivo": "riesgo_p2_t_3h",
            "granularidad": "estacion-hora",
            "n_filas_val": int((tabla_baseline["conjunto"] == "val").sum()),
            "n_filas_test": int((tabla_baseline["conjunto"] == "test").sum()),
        },
        metricas=metricas_baseline,
    )

    del tabla_baseline

    # Entrenar únicamente con train.
    # Reservar val y test para evaluación independiente.
    X_final = X_train
    y_final = y_train

    del tabla

    print("Entrenando modelo Random Forest P2...", flush=True)
    modelo = crear_modelo_p2()
    modelo.fit(X_final, y_final)

    ruta = guardar_modelo_p2(modelo)

    # Evaluación final del Random Forest sobre test.
    metricas_rf = evaluar_modelo(modelo, X_val, y_val, X_test, y_test)

    print(f"Metricas Random Forest P2: {metricas_rf}")

    ruta_matriz = guardar_matriz_confusion(modelo, X_test, y_test)

    print(f"Matriz de confusion guardada en: {ruta_matriz}")

    run_id = registrar_run(
        pregunta="P2",
        run_name="p2-random-forest",
        params={
            "modelo": "RandomForestClassifier",
            "objetivo": "riesgo_p2_t_3h",
            "n_estimators": 50,
            "max_depth": 12,
            "min_samples_leaf": 20,
            "n_filas_entrenamiento": len(X_final),
        },
        metricas=metricas_rf,
        artefactos=[ruta, ruta_matriz],
    )

    print("Entrenamiento completado correctamente.")
    print(f"Árboles: {len(modelo.estimators_)}")
    print(f"Modelo guardado en: {ruta}")
    return run_id


if __name__ == "__main__":
    main()
