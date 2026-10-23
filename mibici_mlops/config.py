import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# --- Proyecto MiBici (Equipo 42): datos, partición y MLflow ---
EQUIPO = "42"
SEED = 42

FECHA_INICIO = "2025-07-01"
VAL_INICIO = "2026-04-01"
TEST_INICIO = "2026-07-01"
FECHA_FIN = "2026-09-30"

# Servidor de tracking (requerido por el Model Registry). Se toma de la variable de entorno,
# por ejemplo MLFLOW_TRACKING_URI=http://127.0.0.1:5000; si no existe, se usa SQLite local.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "mibici-equipo42"

VIAJES_PATH = PROCESSED_DATA_DIR / "viajes"
ESTACIONES_PATH = PROCESSED_DATA_DIR / "estaciones.parquet"
MANIFIESTO_PATH = PROJ_ROOT / "data_manifest.json"

_BASE_URL = "https://mibici.net/site/assets/files"
MESES_URLS = {
    "2025_07": f"{_BASE_URL}/108898/datos_abiertos_2025_07.csv",
    "2025_08": f"{_BASE_URL}/111021/datos_abiertos_2025_08.csv",
    "2025_09": f"{_BASE_URL}/113216/datos_abiertos_2025_09.csv",
    "2025_10": f"{_BASE_URL}/115104/datos_abiertos_2025_10.csv",
    "2025_11": f"{_BASE_URL}/116773/datos_abiertos_2025_11.csv",
    "2025_12": f"{_BASE_URL}/118153/datos_abiertos_2025_12.csv",
    "2026_01": f"{_BASE_URL}/120163/datos_abiertos_2026_01.csv",
    "2026_02": f"{_BASE_URL}/121761/datos_abiertos_2026_02.csv",
    "2026_03": f"{_BASE_URL}/123236/datos_abiertos_2026_03.csv",
    "2026_04": f"{_BASE_URL}/124927/datos_abiertos_2026_04.csv",
    "2026_05": f"{_BASE_URL}/125943/datos_abiertos_2026_05.csv",
    "2026_06": f"{_BASE_URL}/127734/datos_abiertos_2026_06.csv",
    "2026_07": f"{_BASE_URL}/129006/datos_abiertos_2026_07.csv",
    "2026_08": f"{_BASE_URL}/130362/datos_abiertos_2026_08.csv",
    "2026_09": f"{_BASE_URL}/131825/datos_abiertos_2026_09.csv",
}
NOMENCLATURA_URL = f"{_BASE_URL}/1118/nomenclatura_2026_09.csv"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
