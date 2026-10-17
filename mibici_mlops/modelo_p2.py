"""Entrenamiento y evaluación del modelo de clasificación P2 de MiBici."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mibici_mlops.config import PROJ_ROOT

VARIABLES_P2 = [
    "capacidad",
    "hora_dia",
    "dia_semana",
    "es_fin_semana",
    "mes",
    "salidas_lag_1",
    "salidas_lag_2",
    "salidas_lag_24",
    "salidas_lag_168",
    "salidas_media_24h",
    "salidas_media_4sem",
]

OBJETIVO_P2 = "riesgo_p2_t_3h"


def preparar_datos_modelo(tabla: pd.DataFrame):
    """Prepara entrenamiento, validación y prueba para P2."""

    # Excluir registros sin historial suficiente
    tabla = tabla.dropna(subset=VARIABLES_P2 + [OBJETIVO_P2]).copy()

    # Separación cronológica
    train = tabla[tabla["conjunto"] == "train"]
    val = tabla[tabla["conjunto"] == "val"]
    test = tabla[tabla["conjunto"] == "test"]

    X_train = train[VARIABLES_P2].astype("float32")
    y_train = train[OBJETIVO_P2]

    X_val = val[VARIABLES_P2].astype("float32")
    y_val = val[OBJETIVO_P2]

    X_test = test[VARIABLES_P2].astype("float32")
    y_test = test[OBJETIVO_P2]

    return X_train, y_train, X_val, y_val, X_test, y_test


def crear_modelo_p2() -> RandomForestClassifier:
    """Configura el clasificador Random Forest para P2."""

    modelo = RandomForestClassifier(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=20,
        class_weight={0: 1, 1: 10, 2: 10},
        n_jobs=2,
        random_state=42,
    )

    return modelo


def guardar_modelo_p2(modelo: RandomForestClassifier) -> Path:
    """Guarda el modelo P2 entrenado en formato joblib."""

    carpeta = Path(PROJ_ROOT) / "models"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / "modelo_p2.joblib"

    joblib.dump(modelo, ruta)

    print(f"Modelo P2 guardado en: {ruta}")

    return ruta
