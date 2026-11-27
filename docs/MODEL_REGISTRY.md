# Experimentación, Model Registry y linaje — MiBici P1

Este documento junta en un solo lugar cómo se eligió el modelo de P1 que está en producción, cómo está versionado en el Model Registry de MLflow y cómo se puede rastrear desde la predicción de la API hasta los datos y el código con los que se entrenó.

P1: viajes que inician por estación en la siguiente hora (regresión). Modelo registrado: `mibici-equipo42-p1-demanda`.

## 1. Seguimiento de experimentos (MLflow Tracking)

Todas las corridas viven en el experimento `mibici-equipo42` y se lanzan como MLflow Project (`mlflow run . -e train -P question=p1 -P model=<modelo>`). Cada run registra, con el helper común de `mibici_mlops/tracking.py`:

- **Tags:** `equipo`, `question`, `git_commit`, `git_cambios_sin_commit`, `model_family`, `stage_of_work` (`baseline`, `search`, `candidate`, `ab_test`).
- **Parámetros:** partición temporal (`fecha_inicio`, `val_inicio`, `test_inicio`, `fecha_fin`), `data_version` (hash del `data_manifest.json`), `seed` e hiperparámetros.
- **Métricas:** `val_*` para elegir y `test_*` solo para reportar.
- **Artefactos:** modelo con firma y ejemplo de entrada, manifiesto de datos, tabla de entrenamiento agregada (`log_input`), gráficas y configuración.

La tabla completa de runs de P1 se exporta con `python -m scripts.resumen_experimentos_p1` a [`docs/evidencias/actividad11_experimentos_p1.csv`](evidencias/actividad11_experimentos_p1.csv).

## 2. Selección del champion

Partición temporal: train jul-2025 → mar-2026, validación abr → jun-2026, test jul → sep-2026. **Regla:** se elige por `val_mae` (MAE se interpreta directo en viajes por hora); `test` solo se usa para reportar. Todos los candidatos usan el mismo `data_version` (`2a196cfa2823`) y el mismo commit (`d5eeafa`).

| Run | Familia | val MAE | val RMSE | test MAE | test RMSE | Entrenamiento (s) |
|---|---|---:|---:|---:|---:|---:|
| `p1-baseline-semana-anterior` | naive estacional | 0.9933 | 2.0043 | 0.9420 | 1.9212 | — |
| `p1-ridge-a1.0` | ridge | 0.8519 | 1.4868 | 0.8230 | 1.4643 | 5.3 |
| `p1-hgb-squared_error` | hgb | 0.7925 | 1.3837 | 0.7744 | 1.3827 | 24.1 |
| `p1-hgb-poisson` | hgb | **0.7891** | 1.4027 | 0.7732 | 1.4161 | 39.2 |
| `p1-hgb-search` (4 runs hijos) | hgb | — | — | — | — | — |
| `p1-hgb-final` (train + val) | hgb | — | — | **0.7710** | 1.3870 | 62.1 |

Decisiones:

- **HistGradientBoosting sobre Ridge y el baseline:** baja el MAE de validación de 0.9933 (baseline) y 0.8519 (Ridge) a 0.7891, porque captura relaciones no lineales entre hora, día y rezagos.
- **Pérdida Poisson sobre squared_error:** gana en `val_mae` (0.7891 vs 0.7925), que es la métrica de selección, y es la pérdida natural para un conteo con muchos ceros. Squared error tiene un RMSE un poco menor, pero RMSE no es el criterio.
- **Búsqueda de hiperparámetros:** `p1-hgb-search` prueba `learning_rate` ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {31, 63} como runs anidados; gana por `val_mae` la combinación `learning_rate=0.05`, `max_leaf_nodes=63`.
- **Modelo final:** `p1-hgb-final` reentrena esa combinación con train + val y se evalúa una sola vez en test (MAE 0.7710, 18.2 % menos que el baseline).

## 3. Versiones en el Model Registry y estado de producción

Se usan **aliases** (las etapas Staging/Production están obsoletas en MLflow 3). La API siempre carga `models:/mibici-equipo42-p1-demanda@champion`, así que cambiar de modelo es mover el alias, sin tocar código.

| Versión | Run de origen | Datos de entrenamiento | Alias actual | Cómo llegó ahí |
|---|---|---|---|---|
| 1 | `p1-hgb-final` | jul-2025 → jun-2026 (train + val) | `champion_anterior` | Primer champion (`python -m mibici_mlops.registry --question p1`) |
| 2 | `p1-hgb-challenger` (`79607e53c25e457d8affc3684af473ea`) | jul-2025 → ago-2026 | **`champion`** | Promovida por el A/B testing de la semana 9 |

**Promoción de v1 a v2** (detalle en [`AB_TEST_P1.md`](AB_TEST_P1.md)): mismo modelo e hiperparámetros, solo se reentrenó con datos recientes. En sep-2026 (338,690 filas, ningún modelo lo vio) el MAE pasó de 0.8371 a 0.8356 con IC 95 % de la diferencia [−0.0022, −0.0008] y el RMSE de 1.5267 a 1.5212. La regla de promoción se definió antes de correr (límite superior del IC < 0 y RMSE sin empeorar) y se cumplió. La v1 queda con `champion_anterior` para poder regresar con solo mover el alias.

## 4. Linaje de punta a punta

```
data_manifest.json (data_version 2a196cfa2823)
  → run p1-hgb-challenger (params, métricas, git_commit, modelo con firma)
    → Registry del equipo: versión 2, alias champion
      → scripts/exportar_champion.py → models/p1_champion/ + registro_origen.json
        → bootstrap de Compose: run p1-champion-bootstrap + versión con tags de linaje
          → API FastAPI: models:/mibici-equipo42-p1-demanda@champion
```

- `registro_origen.json` guarda el run original con sus parámetros, métricas, tags (`git_commit`, `question`, `stage_of_work`…) y el historial de versiones y aliases del Registry del equipo.
- El bootstrap copia ese linaje al Registry de Compose: el run `p1-champion-bootstrap` lleva los parámetros y métricas originales, y la versión registrada lleva los tags `origen_version`, `origen_run_id`, `origen_run_name`, `git_commit` y `data_version`.
- **Numeración:** el Registry de Compose empieza vacío, así que ahí el champion aparece como **versión 1**, aunque es la **versión 2** del Registry del equipo. Por eso la identidad del modelo se sigue con `origen_version` y `origen_run_id`, no con el número de versión local.

## 5. Verificación automática en CI/CD

El job `compose` de `.github/workflows/ci.yml` levanta el sistema completo y, además de los smoke tests de `/predict`, consulta el alias `champion` en el MLflow de Compose y falla si la versión no trae `origen_run_id` igual al de `registro_origen.json`, o si le faltan `git_commit` o `data_version`. Así, cada cambio a `main` comprueba que el modelo servido sigue siendo trazable. El resultado queda en el resumen del run de GitHub Actions.

## 6. Reproducir

```powershell
# Exportar el champion actual del Registry local (con su linaje)
python -m scripts.exportar_champion
# Tabla de experimentos de P1
python -m scripts.resumen_experimentos_p1
# Levantar el sistema; el bootstrap registra el champion con su linaje
docker compose up -d --build
# Linaje del champion en el MLflow de Compose
curl.exe "http://localhost:5000/api/2.0/mlflow/registered-models/alias?name=mibici-equipo42-p1-demanda&alias=champion"
```
