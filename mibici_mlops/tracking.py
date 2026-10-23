"""Registro común de runs en MLflow para P1, P2 y P3 (Equipo 42).

Aplica en un solo lugar las convenciones del equipo: un experimento por proyecto,
etiquetas de equipo, pregunta y commit, parámetros de reproducción, métricas con
prefijo del conjunto y nada de datos personales.

Dos formas de uso:
- registrar_run(...): entrega 1 (baselines). Abre, llena y cierra un run en una sola llamada.
- iniciar_run(...) + registrar_resultados(...): entrega 2 (candidatos con `mlflow run`).
  Reutiliza el run que crea el MLflow Project, permite runs anidados (búsquedas) y organiza
  los artefactos en subcarpetas: model/, data/, plots/ y config/.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd

from mibici_mlops.config import (
    EQUIPO,
    FECHA_FIN,
    FECHA_INICIO,
    MANIFIESTO_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    SEED,
    TEST_INICIO,
    VAL_INICIO,
)

PREFIJOS_METRICAS = ("train_", "val_", "test_")
CAMPOS_PERSONALES = ("usuario", "genero", "nacimiento")


def obtener_git_commit() -> tuple[str, bool]:
    """Devuelve el hash del commit actual y si hay cambios sin commitear en archivos versionados."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True).strip()
        cambios = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido", True
    return commit, bool(cambios)


def obtener_data_version() -> str:
    """Versión de los datos: primeros 12 caracteres del SHA-256 del manifiesto versionado en Git."""
    if not MANIFIESTO_PATH.exists():
        logger.warning(f"No existe {MANIFIESTO_PATH.name}; se registra data_version='sin-manifiesto'")
        return "sin-manifiesto"
    return hashlib.sha256(MANIFIESTO_PATH.read_bytes()).hexdigest()[:12]


def registrar_run(
    pregunta: str,
    run_name: str,
    params: dict,
    metricas: dict,
    artefactos: list | tuple = (),
    modelo=None,
    tags: dict | None = None,
) -> str:
    """Registra un run en el experimento del equipo y devuelve su run_id.

    pregunta: "P1", "P2" o "P3". run_name: nombre descriptivo (ej. "p1-baseline-semana-anterior").
    params: hiperparámetros y detalles del modelo; se agregan solos fecha_inicio, fecha_fin,
    val_inicio, test_inicio, data_version y seed. metricas: nombres con prefijo train_, val_ o
    test_. artefactos: rutas de archivos (gráficas, mapas). modelo: estimador de scikit-learn opcional.
    """
    malas = [m for m in metricas if not m.startswith(PREFIJOS_METRICAS)]
    if malas:
        raise ValueError(f"Métricas sin prefijo de conjunto (train_/val_/test_): {malas}")
    personales = [p for p in params if any(c in p.lower() for c in CAMPOS_PERSONALES)]
    if personales:
        raise ValueError(f"No se registran datos personales como parámetros: {personales}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    commit, con_cambios = obtener_git_commit()

    params_comunes = {
        "fecha_inicio": FECHA_INICIO,
        "fecha_fin": FECHA_FIN,
        "val_inicio": VAL_INICIO,
        "test_inicio": TEST_INICIO,
        "data_version": obtener_data_version(),
        "seed": SEED,
    }
    etiquetas = {
        "equipo": EQUIPO,
        "pregunta": pregunta,
        "git_commit": commit,
        "git_cambios_sin_commit": str(con_cambios),
        **(tags or {}),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(etiquetas)
        mlflow.log_params({**params_comunes, **params})
        mlflow.log_metrics(metricas)
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta))
        if modelo is not None:
            mlflow.sklearn.log_model(modelo, name="modelo")

    logger.success(f"Run '{run_name}' registrado en '{MLFLOW_EXPERIMENT}': {run.info.run_id}")
    return run.info.run_id

# --------------------------------------------------------------------------- entrega 2
ETAPAS_TRABAJO = ("baseline", "search", "candidate")


def _validar_params(params: dict) -> None:
    personales = [p for p in params if any(c in p.lower() for c in CAMPOS_PERSONALES)]
    if personales:
        raise ValueError(f"No se registran datos personales como parámetros: {personales}")


@contextmanager
def iniciar_run(
    pregunta: str,
    run_name: str,
    model_family: str,
    stage_of_work: str,
    params: dict | None = None,
    nested: bool = False,
    tags: dict | None = None,
):
    """Abre (o reutiliza) un run con las etiquetas y parámetros comunes del equipo.

    - Con `mlflow run`, el proyecto ya creó un run (variable MLFLOW_RUN_ID): se reutiliza en lugar
      de abrir otro, así cada ejecución del proyecto produce exactamente un run de primer nivel.
    - Con nested=True se abre un run hijo del run activo (una combinación de una búsqueda).
    - Etiquetas: equipo, pregunta y question (P1/P2/P3), git_commit, git_cambios_sin_commit,
      model_family y stage_of_work (baseline, search o candidate).
    - Parámetros: fechas de la partición, data_version y seed, más los propios del modelo.
    """
    if stage_of_work not in ETAPAS_TRABAJO:
        raise ValueError(f"stage_of_work debe ser uno de {ETAPAS_TRABAJO}: {stage_of_work}")
    params = params or {}
    _validar_params(params)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.active_run() is None and not nested:
        # Fuera de `mlflow run` (por ejemplo en pruebas locales) se fija el experimento del equipo.
        # Dentro de `mlflow run`, el experimento lo fija el comando (--experiment-name).
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    commit, con_cambios = obtener_git_commit()

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tag("mlflow.runName", run_name)  # el run del proyecto trae un nombre aleatorio
        mlflow.set_tags(
            {
                "equipo": EQUIPO,
                "pregunta": pregunta,
                "question": pregunta,
                "git_commit": commit,
                "git_cambios_sin_commit": str(con_cambios),
                "model_family": model_family,
                "stage_of_work": stage_of_work,
                **(tags or {}),
            }
        )
        mlflow.log_params(
            {
                "fecha_inicio": FECHA_INICIO,
                "fecha_fin": FECHA_FIN,
                "val_inicio": VAL_INICIO,
                "test_inicio": TEST_INICIO,
                "data_version": obtener_data_version(),
                "seed": SEED,
                **params,
            }
        )
        yield run
    logger.success(f"Run '{run_name}' ({stage_of_work}) registrado: {run.info.run_id}")


def registrar_resultados(
    metricas: dict,
    modelo=None,
    X_ejemplo: pd.DataFrame | None = None,
    graficas: list | tuple = (),
    config: dict | None = None,
    datos_entrenamiento: pd.DataFrame | None = None,
    nombre_datos: str | None = None,
    registrar_manifiesto: bool = True,
    tipos_confiables: list[str] | None = None,
) -> str | None:
    """Registra en el run activo las métricas y los artefactos que respaldan el resultado.

    - metricas: con prefijo train_, val_ o test_.
    - modelo + X_ejemplo: modelo de scikit-learn en model/, con firma (signature) inferida de
      X_ejemplo y sus predicciones, y un ejemplo de entrada de 5 filas.
    - graficas: rutas de imágenes → plots/.
    - config: diccionario con la configuración de la corrida → config/run_config.json.
    - datos_entrenamiento: tabla AGREGADA de entrenamiento (nunca viajes ni datos de usuarios);
      se registra con mlflow.log_input, ligada al data_version.
    - registrar_manifiesto: copia data_manifest.json → data/.
    - tipos_confiables: tipos internos del modelo que skops debe aceptar al guardarlo. MLflow 3
      serializa con skops (formato seguro: no ejecuta código al cargar) y rechaza tipos que no
      reconoce; se declaran explícitamente y solo los revisados (p. ej. el TreePredictor de
      HistGradientBoosting).
    Devuelve el model_uri del modelo registrado (o None), para llevarlo al Model Registry.
    """
    malas = [m for m in metricas if not m.startswith(PREFIJOS_METRICAS)]
    if malas:
        raise ValueError(f"Métricas sin prefijo de conjunto (train_/val_/test_): {malas}")
    mlflow.log_metrics(metricas)

    for ruta in graficas:
        mlflow.log_artifact(str(ruta), artifact_path="plots")
    if config is not None:
        mlflow.log_dict(config, "config/run_config.json")
    if registrar_manifiesto and Path(MANIFIESTO_PATH).exists():
        mlflow.log_artifact(str(MANIFIESTO_PATH), artifact_path="data")
    if datos_entrenamiento is not None:
        dataset = mlflow.data.from_pandas(
            datos_entrenamiento,
            name=nombre_datos or "entrenamiento",
            digest=obtener_data_version(),
        )
        mlflow.log_input(dataset, context="training")

    if modelo is None:
        return None
    if X_ejemplo is None:
        raise ValueError("Para registrar el modelo con firma se necesita X_ejemplo")
    muestra = X_ejemplo.head(1000)
    firma = infer_signature(muestra, modelo.predict(muestra))
    info = mlflow.sklearn.log_model(
        modelo,
        name="model",
        signature=firma,
        input_example=X_ejemplo.head(5),
        skops_trusted_types=tipos_confiables,
    )
    mlflow.set_tag("model_uri", info.model_uri)
    return info.model_uri
