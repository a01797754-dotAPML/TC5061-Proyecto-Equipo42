"""Gráficas de interpretación para los experimentos P3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from mibici_mlops.config import FIGURES_DIR


def guardar_mapa_clusters_p3(
    resultado: pd.DataFrame,
    estaciones: pd.DataFrame,
    ruta_salida: str | Path,
) -> Path:
    """Genera el mapa de estaciones coloreadas por cluster.

    La ubicación se utiliza únicamente para interpretar los clusters;
    latitud y longitud no forman parte de las variables del modelo.
    """
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    mapa = resultado.merge(
        estaciones[
            [
                "estacion_id",
                "nombre",
                "latitud",
                "longitud",
            ]
        ],
        on="estacion_id",
        how="left",
    )

    if mapa[["latitud", "longitud"]].isna().any().any():
        raise ValueError(
            "Hay estaciones sin coordenadas para generar el mapa de P3"
        )

    fig, ax = plt.subplots(figsize=(8, 7))

    scatter = ax.scatter(
        mapa["longitud"],
        mapa["latitud"],
        c=mapa["cluster"],
        cmap="tab10",
        s=30,
        alpha=0.8,
    )

    ax.set_title("P3: distribución geográfica de clusters")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    leyenda = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
    )
    ax.add_artist(leyenda)

    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)

    return ruta


def guardar_perfiles_clusters_p3(
    perfil: pd.DataFrame,
    resultado: pd.DataFrame,
    ruta_salida: str | Path,
) -> Path:
    """Grafica el patrón horario promedio de cada cluster.

    Se muestran las proporciones horarias originales para facilitar
    la interpretación de los grupos, no las variables transformadas
    por scaler o PCA.
    """
    ruta = Path(ruta_salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    datos = perfil.merge(
        resultado[["estacion_id", "cluster"]],
        on="estacion_id",
        how="inner",
    )

    if datos.empty:
        raise ValueError(
            "No hay estaciones comparables para generar perfiles P3"
        )

    combinaciones = [
        ("salidas", "semana", "Salidas · entre semana"),
        ("llegadas", "semana", "Llegadas · entre semana"),
        ("salidas", "finde", "Salidas · fin de semana"),
        ("llegadas", "finde", "Llegadas · fin de semana"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharex=True,
    )

    horas = list(range(24))

    for ax, (movimiento, tipo_dia, titulo) in zip(
        axes.flat,
        combinaciones,
        strict=True,
    ):
        columnas = [
            f"{movimiento}_{tipo_dia}_h{hora:02d}"
            for hora in horas
        ]

        faltantes = [
            columna
            for columna in columnas
            if columna not in datos.columns
        ]

        if faltantes:
            raise ValueError(
                f"Faltan columnas para generar perfiles P3: {faltantes}"
            )

        promedios = (
            datos.groupby("cluster")[columnas]
            .mean()
            .sort_index()
        )

        for cluster, fila in promedios.iterrows():
            ax.plot(
                horas,
                fila.to_numpy(),
                marker="o",
                markersize=3,
                label=f"Cluster {cluster}",
            )

        ax.set_title(titulo)
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Proporción promedio")
        ax.set_xticks(range(0, 24, 3))
        ax.legend()

    fig.suptitle(
        "P3: perfiles horarios promedio por cluster",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)

    return ruta


def generar_graficas_p3(
    perfil: pd.DataFrame,
    resultado: pd.DataFrame,
    estaciones: pd.DataFrame,
    nombre_modelo: str,
) -> list[Path]:
    """Genera las gráficas obligatorias de una corrida candidata P3."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nombre_seguro = nombre_modelo.replace("-", "_")

    ruta_mapa = FIGURES_DIR / (
        f"p3_{nombre_seguro}_mapa.png"
    )
    ruta_perfiles = FIGURES_DIR / (
        f"p3_{nombre_seguro}_perfiles.png"
    )

    guardar_mapa_clusters_p3(
        resultado,
        estaciones,
        ruta_mapa,
    )

    guardar_perfiles_clusters_p3(
        perfil,
        resultado,
        ruta_perfiles,
    )

    return [
        ruta_mapa,
        ruta_perfiles,
    ]