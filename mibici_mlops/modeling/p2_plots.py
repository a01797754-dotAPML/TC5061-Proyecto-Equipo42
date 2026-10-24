"""Gráficas de evaluación de los modelos P2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay


def guardar_curva_precision_recall_p2(
    modelo,
    X_val,
    y_val,
    ruta_salida: str | Path,
) -> Path:
    """Genera curvas precisión-recall para desabasto y saturación."""

    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    probabilidades = modelo.predict_proba(X_val)

    fig, ax = plt.subplots(figsize=(8, 6))

    for clase, nombre in [(1, "Desabasto"), (2, "Saturación")]:
        if clase not in modelo.classes_:
            continue

        indice = list(modelo.classes_).index(clase)

        PrecisionRecallDisplay.from_predictions(
            (y_val == clase).astype(int),
            probabilidades[:, indice],
            name=nombre,
            ax=ax,
        )

    ax.set_title("P2 - Curvas precisión-recall (validación)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

    return ruta

def registrar_graficas_p2(
    modelo,
    X_val,
    y_val,
    nombre_modelo: str,
) -> None:
    """Genera y registra las gráficas de validación en el run activo."""
    import tempfile

    import mlflow

    from mibici_mlops.entrenar_p2 import guardar_matriz_predicciones

    if mlflow.active_run() is None:
        raise RuntimeError(
            "Se requiere un run activo para registrar gráficas P2"
        )

    predicciones = modelo.predict(X_val)

    with tempfile.TemporaryDirectory() as carpeta:
        nombre_matriz = f"{nombre_modelo}_matriz_val.png"

        ruta_matriz = guardar_matriz_predicciones(
            y_val,
            predicciones,
            nombre_matriz,
            f"P2 - Matriz de confusión ({nombre_modelo})",
        )

        ruta_curva = guardar_curva_precision_recall_p2(
            modelo,
            X_val,
            y_val,
            Path(carpeta) / f"{nombre_modelo}_precision_recall_val.png",
        )

        mlflow.log_artifact(
            str(ruta_matriz),
            artifact_path="plots",
        )
        mlflow.log_artifact(
            str(ruta_curva),
            artifact_path="plots",
        )

        print(
            f"Gráficas P2 registradas: {nombre_modelo}",
            flush=True,
        )
