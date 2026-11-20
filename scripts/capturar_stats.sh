#!/usr/bin/env bash
# Muestrea CPU y memoria de los contenedores durante una prueba de carga.
# Uso: scripts/capturar_stats.sh <salida.csv> [docker|podman] [intervalo_s] [contenedor...]
# Se detiene con Ctrl+C.
#
# Ojo: `podman stats --no-stream` reporta el CPU promedio desde que arrancó el contenedor,
# no el instantáneo. Por eso con podman se usa el modo continuo (--interval), que sí da
# el CPU del intervalo. `docker stats --no-stream` ya mide el instantáneo.
set -euo pipefail
OUT=${1:-reports/locust/stats.csv}
ENGINE=${2:-docker}
INTERVAL=${3:-2}
shift $(( $# < 3 ? $# : 3 ))
FMT='{{.Name}},{{.CPUPerc}},{{.MemUsage}}'
mkdir -p "$(dirname "$OUT")"
echo "timestamp,name,cpu_perc,mem_usage" > "$OUT"
if [ "$ENGINE" = "podman" ]; then
  podman stats --interval "$INTERVAL" --format "$FMT" "$@" \
    | while IFS= read -r linea; do
        case "$linea" in *%*) echo "$(date +%s),$linea" >> "$OUT" ;; esac
      done
else
  while true; do
    ts=$(date +%s)
    docker stats --no-stream --format "$FMT" "$@" | sed "s/^/$ts,/" >> "$OUT"
    sleep "$INTERVAL"
  done
fi
