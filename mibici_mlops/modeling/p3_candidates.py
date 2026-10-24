"""Modelos candidatos para P3: clustering de estaciones MiBici.

Todos los modelos se construyen como pipelines reproducibles para que
el preprocesamiento viaje junto con KMeans al registrarlo en MLflow.
"""

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici_mlops.config import SEED

FLUJO_COL = "flujo_neto_rel_capacidad"

CANDIDATOS_P3 = (
    "baseline",
    "kmeans-sin-escalar",
    "kmeans-flujo-escalado",
    "pca-flujo-escalado",
)


def _kmeans(k: int) -> KMeans:
    """Construye KMeans con la configuración reproducible del equipo."""
    return KMeans(
        n_clusters=k,
        random_state=SEED,
        n_init=20,
    )


def crear_candidato_p3(
    nombre: str,
    columnas: list[str],
    params: dict | None = None,
) -> Pipeline:
    """Construye un pipeline candidato de clustering para P3.

    Modelos disponibles
    -------------------
    baseline
        StandardScaler sobre todas las variables y después KMeans.
        Reproduce la lógica del baseline de la Entrega 1.

    kmeans-sin-escalar
        KMeans directamente sobre las proporciones horarias y flujo neto.

    kmeans-flujo-escalado
        Conserva las proporciones horarias en su escala natural [0, 1]
        y estandariza únicamente flujo_neto_rel_capacidad.

    pca-flujo-escalado
        Estandariza únicamente flujo_neto_rel_capacidad, aplica PCA y
        finalmente KMeans.

    Parámetros reconocidos
    ----------------------
    k:
        Número de clusters. Por defecto 3.

    pca_varianza:
        Proporción de varianza que debe conservar PCA. Por defecto 0.90.
    """
    if nombre not in CANDIDATOS_P3:
        raise ValueError(
            f"Candidato P3 desconocido: {nombre}. "
            f"Opciones: {CANDIDATOS_P3}"
        )

    params = params or {}

    k = int(params.get("k", 3))
    pca_varianza = float(params.get("pca_varianza", 0.90))

    if k < 2:
        raise ValueError("P3 requiere al menos 2 clusters")

    if not 0 < pca_varianza <= 1:
        raise ValueError("pca_varianza debe estar entre 0 y 1")

    if FLUJO_COL not in columnas:
        raise ValueError(
            f"Falta la variable requerida de P3: {FLUJO_COL}"
        )

    if nombre == "baseline":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("kmeans", _kmeans(k)),
            ]
        )

    if nombre == "kmeans-sin-escalar":
        return Pipeline(
            steps=[
                ("kmeans", _kmeans(k)),
            ]
        )

    preprocesador_flujo = ColumnTransformer(
        transformers=[
            (
                "flujo",
                StandardScaler(),
                [FLUJO_COL],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    if nombre == "kmeans-flujo-escalado":
        return Pipeline(
            steps=[
                ("preprocesador", preprocesador_flujo),
                ("kmeans", _kmeans(k)),
            ]
        )

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador_flujo),
            (
                "pca",
                PCA(
                    n_components=pca_varianza,
                    svd_solver="full",
                ),
            ),
            ("kmeans", _kmeans(k)),
        ]
    )