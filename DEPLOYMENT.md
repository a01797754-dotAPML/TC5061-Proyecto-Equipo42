# Despliegue de MiBici con Docker

## Construcción
```bash
docker build -f Containerfile -t mibici-equipo42-api:dev .
```

## Ejecución
```bash
docker run --rm --name mibici-api -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 mibici-equipo42-api:dev
```

## Arquitectura
La API FastAPI se ejecuta en un contenedor Docker y consulta los modelos registrados en un servidor MLflow externo mediante MLFLOW_TRACKING_URI.

## Estado de validación
Imagen Docker construida correctamente. Importaciones de FastAPI, Uvicorn, MLflow y mibici_mlops verificadas.
Pendiente: integración de la API, arranque del servidor y pruebas HTTP.
