# Despliegue de MiBici con Docker

Las instrucciones completas (registro del modelo, Podman, ejemplos de petición y problemas conocidos) están en la sección **Actividad 8** del `README.md`.

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
Imagen construida y probada de punta a punta con Docker (y construida con Podman): el contenedor carga `models:/mibici-equipo42-p1-demanda@champion` desde el Registry y responde `/health` (200), `/predict` (200) y entradas inválidas (422). Evidencia en `docs/evidencias/actividad8_*.png`.

## Configuración del servidor MLflow

Para permitir conexiones desde el contenedor Docker, iniciar MLflow
en la máquina anfitriona con los parámetros de red adecuados:

    python -m mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --host 0.0.0.0 \
      --port 5000 \
      --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*,host.containers.internal:*"

La lista de `--allowed-hosts` va separada por comas y cada host lleva `:*` (cualquier puerto),
porque MLflow compara contra el encabezado `Host` completo (por ejemplo `host.docker.internal:5000`).
Sin el puerto o con otro separador, MLflow responde 403 a todas las peticiones, incluso desde localhost.

El contenedor utiliza la variable de entorno:

    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

La configuración de allowed-hosts permite que MLflow acepte
solicitudes desde Docker y evita errores HTTP 403 por validación
del encabezado Host.

Nota de seguridad: utilizar esta configuración únicamente en un
entorno de desarrollo confiable. No exponer el puerto 5000 a Internet.

Para registrar el modelo champion de P1:

    python -m mibici_mlops.registry --question p1

## Actividad 10 - Orquestación con Docker Compose

El archivo `compose.yaml` permite iniciar la API FastAPI y el
servidor MLflow mediante un solo comando.

### Requisitos

- Docker Desktop o Docker Engine con Docker Compose.
- Puertos locales 8000 y 5000 disponibles.

### Construcción e inicio

```bash
docker compose up -d --build
