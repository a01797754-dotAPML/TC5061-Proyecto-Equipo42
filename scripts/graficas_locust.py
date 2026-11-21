"""Gráficas de una corrida de Locust + muestreo de stats del contenedor.

Uso:
    python scripts/graficas_locust.py reports/locust/run1 reports/locust/stats_run1.csv docs/evidencias/actividad10_run1
Lee <prefijo>_stats_history.csv (de `locust --csv <prefijo>`) y el CSV de capturar_stats.sh.
"""

import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

UNIDADES = {"b": 1e-6, "kb": 1e-3, "kib": 1.024e-3, "mb": 1, "mib": 1.048576, "gb": 1e3, "gib": 1073.741824}


def a_mb(texto: str) -> float:
    valor, unidad = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", texto.split("/")[0]).groups()
    return float(valor) * UNIDADES[unidad.lower()]


def main(prefijo: str, stats_csv: str, salida: str) -> None:
    h = pd.read_csv(f"{prefijo}_stats_history.csv")
    h = h[h["Name"] == "Aggregated"].copy()
    h["t"] = h["Timestamp"] - h["Timestamp"].iloc[0]
    h["error_pct"] = 100 * h["Failures/s"] / h["Requests/s"].where(h["Requests/s"] > 0)

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(h["t"], h["50%"], label="p50")
    ax[0].plot(h["t"], h["95%"], label="p95")
    ax[0].set_ylabel("latencia (ms)")
    ax[0].legend()
    ax[1].plot(h["t"], h["Requests/s"], label="RPS")
    ax1b = ax[1].twinx()
    ax1b.plot(h["t"], h["User Count"], color="gray", ls="--", label="usuarios")
    ax[1].set_ylabel("RPS")
    ax1b.set_ylabel("usuarios")
    ax[2].plot(h["t"], h["error_pct"].fillna(0), color="tab:red")
    ax[2].set_ylabel("errores (%)")
    ax[2].set_xlabel("tiempo (s)")
    fig.tight_layout()
    fig.savefig(f"{salida}_locust.png", dpi=150)

    s = pd.read_csv(stats_csv)
    # La primera muestra de cada contenedor es el promedio acumulado, no el instantáneo.
    s = s[s.groupby("name").cumcount() > 0].copy()
    s["t"] = s["timestamp"] - h["Timestamp"].iloc[0]
    s["cpu"] = s["cpu_perc"].str.rstrip("%").astype(float)
    s["mem_mb"] = s["mem_usage"].map(a_mb)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for nombre, g in s.groupby("name"):
        ax[0].plot(g["t"], g["cpu"], label=nombre)
        ax[1].plot(g["t"], g["mem_mb"], label=nombre)
    ax[0].set_ylabel("CPU (%)")
    ax[0].legend()
    ax[1].set_ylabel("memoria (MB)")
    ax[1].set_xlabel("tiempo (s)")
    fig.tight_layout()
    fig.savefig(f"{salida}_recursos.png", dpi=150)

    resumen = s.groupby("name").agg(cpu_media=("cpu", "mean"), cpu_pico=("cpu", "max"), mem_pico_mb=("mem_mb", "max"))
    print(resumen.round(1).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
