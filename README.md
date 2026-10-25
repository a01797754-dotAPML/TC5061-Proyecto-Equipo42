# MiBici MLOps · TC5061 Equipo 42

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Pipeline reproducible de Machine Learning sobre los datos abiertos de **MiBici**, el sistema de bicicleta pública del Área Metropolitana de Guadalajara. Primera entrega del proyecto integrador de TC5061 Operaciones de Aprendizaje Automático (Tecnológico de Monterrey).

## Propósito

Responder con Machine Learning tres preguntas sobre la operación de MiBici y dejar una base reproducible: ingesta versionada, variables sin información del futuro, baselines registrados en MLflow y un flujo de trabajo en equipo con Pull Requests. La propuesta completa está en el ML Canvas ([`references/ml_canvas.md`](references/ml_canvas.md)) y la arquitectura en [`references/arquitectura.png`](references/arquitectura.png).

## Preguntas del proyecto

| # | Pregunta | Tarea | Formulación | Usuario |
|---|---|---|---|---|
| P1 | ¿Cuántos viajes iniciarán en una estación en la siguiente hora? | Regresión | Salidas por estación × hora, horizonte 1 h | Operador del sistema |
| P2 | ¿La estación se quedará sin bicis o sin anclajes? | Clasificación (3 clases) | Flujo neto de la hora / capacidad: < −0.30 desabasto, > +0.30 saturación; horizonte 1 h | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso? | Clustering | Perfil horario de salidas y llegadas (entre semana y fin de semana) + flujo neto relativo | Planeación urbana |

## Datos

- **Fuente:** [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/): viajes mensuales en CSV y nomenclatura de estaciones (`nomenclatura_2026_09.csv`). Las URLs exactas de cada mes están en `mibici_mlops/config.py`.
- **Rango:** 15 meses, de julio de 2025 a septiembre de 2026 → **5,616,821 viajes** (tras eliminar 758 duplicados) y **484 estaciones**.
- **Partición temporal:** entrenamiento jul-2025 a mar-2026 · validación abr a jun-2026 · test jul a sep-2026 (fechas de corte en `config.py`).
- **Versionado:** `data_manifest.json` (en Git) guarda número de registros y SHA-256 de cada archivo crudo y procesado; su hash es el `data_version` que se registra en MLflow.
- **Datos personales:** la ingesta solo lee 5 columnas (viaje, inicio, fin, origen, destino); `Usuario_Id`, `Genero` y `Año_de_nacimiento` nunca se leen ni se guardan.
- **Uso:** el portal no publica una licencia abierta explícita; los datos se usan con fines académicos y **no se redistribuyen** (están en `.gitignore`; se descargan con `make data`).

## Estructura del repositorio

```
├── MLproject                <- MLflow Project: puntos de entrada ingest y train
├── python_env.yaml          <- Entorno del MLflow Project (mismas versiones que requirements.txt)
├── Makefile                 <- make requirements | data | experiments | test | lint
├── README.md
├── data_manifest.json       <- Versión de los datos (registros + SHA-256 por archivo)
├── pyproject.toml           <- Metadatos del paquete mibici_mlops y configuración de ruff
├── requirements.txt         <- Dependencias con versión fija (Python 3.12)
├── data/                    <- Fuera de Git: raw/ (CSV originales) y processed/ (Parquet)
├── docs/                    <- Proyecto mkdocs de la plantilla
├── models/                  <- Modelos serializados (fuera de Git)
├── notebooks/               <- Solo exploración: 1.0-ce-exploracion-mibici.ipynb
├── references/              <- ML Canvas (md, png, html) y diagrama de arquitectura (png, dot)
├── reports/figures/         <- Figuras generadas por los experimentos (fuera de Git)
├── tests/                   <- Pruebas de ingesta, features de P3, modelo P2 y registro común / MLflow Project (pytest)
└── mibici_mlops/            <- Código fuente
    ├── config.py            <- Rutas, fechas, partición, semilla, URLs y MLflow (un solo lugar)
    ├── dataset.py           <- Ingesta: descarga, normalización y Parquet
    ├── manifest.py          <- Genera data_manifest.json
    ├── features.py          <- Tabla estación × hora, calendario, rezagos, etiqueta P2, perfiles P3
    ├── tracking.py          <- Registro común en MLflow: registrar_run (baselines), iniciar_run + registrar_resultados (candidatos)
    ├── preparar_p2.py, modelo_p2.py, entrenar_p2.py   <- Preparación, modelo y entrenamiento de P2
    └── modeling/
        ├── p1.py, p2.py, p3.py      <- main() (baseline) y train(model, params) (candidatos, entrega 2)
        ├── train.py                 <- Punto de entrada train del MLflow Project
        └── run_experiments.py       <- Orquesta P1, P2 y P3 (make experiments)
```

## Instalación

Requiere Python 3.12, `make` y `curl`.

```bash
git clone https://github.com/a01797754-dotAPML/TC5061-Proyecto-Equipo42.git
cd TC5061-Proyecto-Equipo42
conda create -n mibici python=3.12 -y && conda activate mibici   # o un venv con Python 3.12
make requirements                                                # instala dependencias fijas + el paquete
```

## Ingesta de datos

```bash
make data
```

Descarga los 15 meses y la nomenclatura, normaliza codificación, columnas y fechas, elimina duplicados, guarda Parquet en `data/processed/` y regenera `data_manifest.json`. Desde un clon limpio, el manifiesto resulta idéntico al versionado (`data_version = 2a196cfa2823`).

## Experimento base

```bash
make experiments                                                # P1, P2 y P3 → experimento mibici-equipo42
python -m mibici_mlops.modeling.run_experiments --pregunta P2   # una sola pregunta (repetible)
python -c "from mibici_mlops.modeling import p3; p3.main_pca()" # candidato P3: PCA + KMeans
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db     # interfaz en http://127.0.0.1:5000
```

`make experiments` se niega a correr con cambios sin commitear (salvo `--permitir-cambios`), sigue con las demás preguntas si una falla y termina con código 1 si hubo errores. Usar `python -m mlflow` garantiza la versión de MLflow del entorno.

| Run | Métricas de test |
|---|---|
| `p1-baseline-semana-anterior` | MAE 0.942 · RMSE 1.921 |
| `p2-baseline-semana-anterior` | macro-F1 0.473 · recall desabasto 0.182 · recall saturación 0.243 |
| `p2-random-forest` | macro-F1 0.512 · recall desabasto 0.382 · recall saturación 0.458 |
| `p3-baseline-kmeans` | silhouette 0.074 · Davies-Bouldin 3.05 · ARI 0.686 |
| `p3-pca-kmeans` | silhouette 0.095 · Davies-Bouldin 2.72 · ARI 0.633 |

Pruebas: `make test` (o `python -m pytest tests`).

## MLflow Project (entrega 2)

El repositorio es un **MLflow Project** (`MLproject` en la raíz): la ingesta y el entrenamiento de P1, P2 y P3 se ejecutan con `mlflow run`. El entorno se declara en `python_env.yaml` con las mismas versiones fijas de `requirements.txt`; `mlflow run` lo crea con **pyenv** (`brew install pyenv && pyenv install 3.12.15`).

### 1. Servidor de tracking (requerido por el Model Registry)

En una terminal aparte, desde la raíz del repo:

```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Y en la terminal de trabajo:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

`config.py` toma `MLFLOW_TRACKING_URI` de esta variable (si no existe, usa `sqlite:///mlflow.db`). `mlflow.db`, `mlruns/` y `mlartifacts/` están en `.gitignore`. Usar `python -m mlflow` garantiza la versión de MLflow del entorno.

### 2. Ingesta y entrenamiento con `mlflow run`

```bash
# Ingesta: descarga, normaliza y versiona los datos (dataset.py + manifest.py)
python -m mlflow run . -e ingest --experiment-name mibici-equipo42

# Entrenamiento de un candidato: pregunta, modelo e hiperparámetros (JSON)
python -m mlflow run . -e train --experiment-name mibici-equipo42 \
    -P question=<p1|p2|p3> -P model=<modelo> -P params='{"<hiperparametro>": <valor>}'
```

| Parámetro | Valores | Por defecto |
|---|---|---|
| `question` | `p1`, `p2`, `p3` | `p1` |
| `model` | Los que implementa cada `modeling/pN.py` | Obligatorio |
| `params` | Hiperparámetros en JSON | `{}` (valores por defecto del modelo) |

Rutas, fechas, partición y semilla salen de `config.py`; no se pasan por parámetro. Para probar sin MLflow Project: `python -m mibici_mlops.modeling.train --question <p1|p2|p3> --model <modelo>`.

**Contrato por pregunta:** cada `mibici_mlops/modeling/pN.py` expone `train(model: str, params: dict) -> str` (devuelve el `run_id`) y registra la corrida con `tracking.iniciar_run` + `tracking.registrar_resultados`. El ejemplo de uso completo está en `tests/test_mlflow_project.py`.

### 3. Qué registra cada candidato

Cada corrida candidata reutiliza el run que crea `mlflow run` y registra, con el helper común `tracking.iniciar_run` + `tracking.registrar_resultados`:

| Elemento | Dónde |
|---|---|
| Etiquetas `equipo`, `question`, `git_commit`, `git_cambios_sin_commit`, `model_family`, `stage_of_work` (`baseline`, `search`, `candidate`) | Tags del run |
| Fechas de la partición, `data_version`, `seed` e hiperparámetros | Parámetros del run |
| Métricas `val_*` y `test_*` (las mismas de la entrega 1) y la mejora contra el baseline | Métricas del run |
| Modelo con firma (`signature`) y ejemplo de entrada | *Logged model* `model`, ligado al run |
| Manifiesto de datos | `data/data_manifest.json` |
| Conjunto de entrenamiento **agregado** (estación × hora) | Input del run (`mlflow.log_input`) |
| Gráficas de desempeño | `plots/` |
| Configuración de la corrida | `config/run_config.json` |

Las búsquedas de hiperparámetros se organizan como **nested runs**: un run padre (`stage_of_work=search`) y un run hijo por combinación (`iniciar_run(..., nested=True)`).

### 4. Model Registry

Esquema de ciclo de vida: **aliases** (el esquema vigente de MLflow; las etapas *Staging/Production* están obsoletas). `champion` marca la versión en producción y `challenger` la siguiente mejor. Modelos:

| Pregunta | Modelo registrado | Carga en producción |
|---|---|---|
| P1 | `mibici-equipo42-p1-demanda` | `models:/mibici-equipo42-p1-demanda@champion` |
| P2 | `mibici-equipo42-p2-riesgo` | `models:/mibici-equipo42-p2-riesgo@champion` |
| P3 | `mibici-equipo42-p3-grupos` | `models:/mibici-equipo42-p3-grupos@champion` |

Al registrar cada versión se copian `git_commit` y `data_version` del run como tags de la versión, para que el linaje se vea en la página del modelo.

### 5. Reproducir la experimentación completa (corrida oficial)

Con el servidor de tracking arriba y `MLFLOW_TRACKING_URI` apuntando a él, desde un clon limpio. El **orden importa** en P1: `hgb-final` toma la mejor combinación de la última `p1-hgb-search`.

```bash
python -m mlflow run . -e ingest --experiment-name mibici-equipo42

for m in baseline ridge hgb hgb-squared hgb-search hgb-final; do
  python -m mlflow run . -e train --experiment-name mibici-equipo42 -P question=p1 -P model=$m
done
for m in baseline rf-ajustado-1 rf-ajustado-2 rf-ajustado-3 rf-search; do
  python -m mlflow run . -e train --experiment-name mibici-equipo42 -P question=p2 -P model=$m
done
for m in baseline kmeans-sin-escalar kmeans-flujo-escalado pca-flujo-escalado kmeans-search; do
  python -m mlflow run . -e train --experiment-name mibici-equipo42 -P question=p3 -P model=$m
done
```

Ninguna corrida necesita `params`: los valores por defecto y las mallas de búsqueda están en cada `modeling/pN*.py`.

**Windows:** `mlflow run` escapa dos veces las comillas al pasar el comando a `cmd`, así que cualquier `-P params` con JSON falla (`Invalid \escape`). Usar `--env-manager local` y los comandos sin `params`.

**Cachés:** P1 (`data/interim`) y P2 (`data/processed/p2_cache`) guardan sus tablas ligadas al `data_version`. Si cambia el código de features sin cambiar los datos, borrar la caché antes de correr.

## Buenas prácticas aplicadas

| Práctica | Cómo la aplicamos |
|---|---|
| Un experimento, runs con nombre descriptivo | Experimento `mibici-equipo42`; nombres como `p2-random-forest`, nunca aleatorios |
| Parámetros para reproducir | `registrar_run` agrega siempre fechas, `val_inicio`, `test_inicio`, `data_version` y `seed`; cada pregunta agrega su modelo e hiperparámetros |
| Run ligado a código y pregunta | Tags `equipo`, `pregunta`, `git_commit` y `git_cambios_sin_commit` |
| Métricas con el conjunto en el nombre | `registrar_run` rechaza métricas sin prefijo `train_`, `val_` o `test_` |
| Sin credenciales ni datos personales | La ingesta no lee columnas de usuario; `registrar_run` rechaza parámetros con esos nombres; `.env` en `.gitignore` |
| Notebooks solo para exploración | La lógica vive en `mibici_mlops/`; el notebook sigue la convención `1.0-iniciales-descripcion` |
| Configuración en un solo lugar | `config.py` (rutas, fechas, semilla, URLs, MLflow) |
| Dependencias fijas y entorno reproducible | `requirements.txt` con versiones fijas, verificado en un entorno nuevo desde un clon limpio |
| Nada de datos, modelos ni MLflow en Git | `.gitignore` excluye `data/`, `models/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `.env` y salidas de `reports/` |
| Cambios a `main` solo por PR revisado | Rama `main` protegida (requiere aprobación) |

## Flujo de trabajo con Git (GitHub Flow)

1. `main` está protegida: nadie hace push directo; todo entra por Pull Request con al menos una aprobación.
2. Cada cambio en su rama, creada desde `main` actualizado: `tipo/descripcion-corta` (por ejemplo `feat/p3-pca-kmeans`, `docs/canvas-p2-visual`, `chore/requirements-versiones`).
3. Commits con [Conventional Commits](https://www.conventionalcommits.org/): `feat`, `fix`, `docs`, `chore`, `refactor`, `test` con alcance opcional, por ejemplo `fix(p2): registrar la matriz de confusión del baseline`.
4. Antes de pedir revisión se trae `main` a la rama y se resuelven conflictos en la rama, no en `main`.
5. Se hace `git status` antes de cada `git add` para no subir datos, modelos ni `mlflow.db`.

## Integrantes y roles

| Integrante | Matrícula | Rol |
|---|---|---|
| Carlos Eduardo Medina Flores | A01753742 | ML Canvas y formulación de P1–P3, pipeline y arquitectura; tabla estación × hora en `features.py`, `tracking.py` (`registrar_run`) y baseline de P1; notebook de exploración |
| Víctor Manuel Martínez Zarate | A01840404 | Ingesta reproducible (`dataset.py`, `manifest.py`) y versionado SHA-256; P2: etiqueta de riesgo, variables (incluidas tasas históricas), baseline, Random Forest y su registro en MLflow; pruebas automatizadas de ingesta y P2; propuesta del runner por pregunta (`--pregunta`) integrado en `run_experiments` |
| Andrea Paola Martínez López | A01797754 | Repositorio con Cookiecutter Data Science, `.gitignore` y GitHub Flow; P3: perfiles de uso, KMeans y candidato PCA + KMeans |
| Carlos David Martínez Calva | A01056864 | `run_experiments` y `make experiments`, corrida oficial desde un clon limpio y evidencia de MLflow; revisión de PRs; integración del documento de la propuesta y README |
