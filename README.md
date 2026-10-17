# MiBici MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Pipeline reproducible de Machine Learning para el análisis de datos de MiBici

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_mlops and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici_mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Modelo P2: clasificación de riesgo de estaciones MiBici

### Objetivo

Desarrollar un modelo de clasificación que identifique tres categorías de riesgo a partir de los movimientos históricos de bicicletas por estación y hora:

- **Clase 0:** bajo riesgo.
- **Clase 1:** riesgo de desabasto.
- **Clase 2:** riesgo de saturación.

Las etiquetas se construyen mediante una heurística basada en los flujos horarios y la capacidad registrada de las estaciones. No representan mediciones directas de la ocupación real.

### Preparación de datos

El pipeline procesa 15 meses de datos de MiBici, consolida los movimientos por estación y hora, completa las horas sin movimientos e incorpora la capacidad de cada estación.

Se generan variables de calendario, rezagos históricos y promedios móviles. Los registros sin suficiente historial para calcular las características se excluyen del entrenamiento.

La separación temporal utiliza los siguientes periodos:

- **Train:** julio de 2025 a marzo de 2026.
- **Validation:** abril a junio de 2026.
- **Test:** julio a septiembre de 2026.

### Entrenamiento

Para ejecutar el entrenamiento reproducible:

```bash
python -m mibici_mlops.entrenar_p2
```

El modelo utiliza Random Forest con 50 árboles, profundidad máxima de 12, un mínimo de 20 muestras por hoja y pesos personalizados para las clases `{0: 1, 1: 10, 2: 10}`.

El artefacto se guarda en `models/modelo_p2.joblib`. Este archivo está excluido del repositorio mediante `.gitignore`; debe generarse localmente ejecutando el entrenamiento.

### Evaluación

El modelo seleccionado obtuvo los siguientes resultados en el conjunto de prueba:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.97 |
| Macro-F1 | 0.46 |
| Recall de desabasto | 0.30 |
| Recall de saturación | 0.19 |

Las métricas detalladas y la matriz de confusión están registradas en `reports/metrics/metricas_p2.json`.

### Pruebas automatizadas

Para ejecutar las pruebas específicas de P2:

```bash
python -m pytest tests/test_modelo_p2.py -v
```

Las pruebas verifican las variables predictoras, los parámetros de Random Forest y el guardado y recuperación del modelo.

### Limitaciones

El dataset presenta un fuerte desbalance de clases. Aunque la exactitud global es alta, el recall de las clases minoritarias continúa siendo limitado. Por ello, el modelo debe considerarse una primera versión experimental y no un sistema operativo de alertas.

Las etiquetas son aproximaciones heurísticas derivadas de los flujos y la capacidad registrada; no se ha demostrado que correspondan directamente a la disponibilidad real de bicicletas.
