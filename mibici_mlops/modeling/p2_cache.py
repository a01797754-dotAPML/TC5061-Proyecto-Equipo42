"""Caché local de particiones para experimentos P2."""

import json

import pandas as pd

from mibici_mlops.config import PROJ_ROOT
from mibici_mlops.tracking import obtener_data_version

CACHE_DIR = PROJ_ROOT / "data" / "processed" / "p2_cache"

NOMBRES = (
    "X_train",
    "y_train",
    "X_val",
    "y_val",
    "X_test",
    "y_test",
)


def cargar_o_preparar_p2():
    """Reutiliza particiones si corresponden a la versión de datos actual."""
    version = obtener_data_version()
    metadata_path = CACHE_DIR / "metadata.json"

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        archivos = [
            CACHE_DIR / f"{nombre}.parquet"
            for nombre in NOMBRES
        ]

        if (
            metadata.get("data_version") == version
            and all(archivo.exists() for archivo in archivos)
        ):
            print(f"Reutilizando caché P2: {version}")
            partes = [
                pd.read_parquet(archivo)
                for archivo in archivos
            ]
            return tuple(
                parte.iloc[:, 0] if nombre.startswith("y_") else parte
                for nombre, parte in zip(NOMBRES, partes)
            )

    print(f"Preparando caché P2: {version}")

    from mibici_mlops.modelo_p2 import preparar_datos_modelo
    from mibici_mlops.preparar_p2 import generar_dataset_p2

    tabla = generar_dataset_p2()
    partes = preparar_datos_modelo(tabla)
    del tabla

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for nombre, parte in zip(NOMBRES, partes):
        archivo = CACHE_DIR / f"{nombre}.parquet"
        if isinstance(parte, pd.Series):
            parte.to_frame(name=nombre).to_parquet(archivo, index=False)
        else:
            parte.to_parquet(archivo, index=False)

    metadata_path.write_text(
        json.dumps({"data_version": version}, indent=2)
    )

    print("Caché P2 guardada correctamente.")
    return partes
