"""Configuraciones de modelos candidatos para P2."""

from sklearn.ensemble import RandomForestClassifier

from mibici_mlops.config import SEED

CANDIDATOS_P2 = {
    "rf-ajustado-1": {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_leaf": 20,
    },
    "rf-ajustado-2": {
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_leaf": 10,
    },
    "rf-ajustado-3": {
        "n_estimators": 150,
        "max_depth": 16,
        "min_samples_leaf": 20,
    },
}

def crear_candidato_p2(
    nombre: str,
    params: dict | None = None,
) -> RandomForestClassifier:
    """Construye un candidato P2 con hiperparámetros configurables."""

    if nombre not in CANDIDATOS_P2:
        raise ValueError(f"Candidato P2 desconocido: {nombre}")

    configuracion = CANDIDATOS_P2[nombre].copy()

    if params:
        configuracion.update(params)

    return RandomForestClassifier(
        **configuracion,
        class_weight={0: 1, 1: 10, 2: 10},
        n_jobs=2,
        random_state=SEED,
    )
