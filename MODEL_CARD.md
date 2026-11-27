# Model Card — MiBici P1: demanda por estación en la siguiente hora

Ficha del modelo que está en producción, siguiendo la estructura de Mitchell et al. (2019). El detalle de la selección, las versiones y el linaje está en [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md).

## Detalles del modelo

| Característica | Descripción |
|---|---|
| Nombre en el Registry | `mibici-equipo42-p1-demanda` |
| Versión servida | Alias `champion` → versión 2 del Registry del equipo (run `79607e53c25e457d8affc3684af473ea`, `p1-hgb-challenger`) |
| Algoritmo | `HistGradientBoostingRegressor` con pérdida Poisson (`learning_rate=0.05`, `max_leaf_nodes=63`, `seed=42`) |
| Tipo de aprendizaje | Supervisado, regresión |
| Serialización | skops (carga sin ejecutar código arbitrario, a diferencia de pickle) |
| Servicio | API FastAPI (`POST /predict`) en Docker Compose; carga `models:/mibici-equipo42-p1-demanda@champion` |
| Responsables | Equipo 42, TC5061 Operaciones de Aprendizaje Automático (Tecnológico de Monterrey) |
| Fecha | Noviembre de 2026 |

## Uso previsto

- **Uso principal:** estimar cuántos viajes iniciarán en cada estación de MiBici en la siguiente hora, como apoyo para planear el rebalanceo de bicicletas.
- **Usuarios previstos:** el equipo de operación del sistema.
- **Fuera de alcance:** no sustituye la supervisión operativa, no predice llegadas ni demanda a más de una hora y no está pensado para estaciones nuevas sin historial.

## Datos

- **Fuente:** datos abiertos de viajes de MiBici (Área Metropolitana de Guadalajara), usados con fines académicos y sin redistribuirse.
- **Entrenamiento del modelo servido:** julio de 2025 a agosto de 2026 (`data_version 2a196cfa2823`, hash del `data_manifest.json`).
- **Variables (11):** calendario (`hora_dia`, `dia_semana`, `es_fin_semana`, `mes`), `capacidad` de la estación, rezagos de salidas a 1, 2, 24 y 168 horas, y medias de las últimas 24 horas y de la misma hora en las 4 semanas previas. Todas se calculan solo con información del pasado.
- **Privacidad:** la ingesta nunca lee `Usuario_Id`, género ni año de nacimiento.

## Desempeño

| Evaluación | Datos | MAE | RMSE |
|---|---|---:|---:|
| Selección del modelo (`p1-hgb-final`, versión 1) | Test jul–sep 2026 | 0.7710 | 1.3870 |
| Baseline (salidas de la misma hora de la semana anterior) | Test jul–sep 2026 | 0.9420 | 1.9212 |
| A/B offline: versión 1 (champion anterior) | Sep 2026, 338,690 filas | 0.8371 | 1.5267 |
| A/B offline: **versión 2 (champion actual)** | Sep 2026, 338,690 filas | **0.8356** | **1.5212** |

La versión 2 se promovió porque el intervalo de confianza al 95 % de la diferencia de MAE quedó completo debajo de cero ([−0.0022, −0.0008]) y el RMSE no empeoró, que era la regla definida antes del A/B ([`docs/AB_TEST_P1.md`](docs/AB_TEST_P1.md)). MAE se interpreta en viajes por hora. No se usa MAPE porque el 53.1 % de las estación-horas tienen 0 salidas.

## Limitaciones y riesgos

- **Estaciones nuevas:** en los primeros 28 días de historial, `salidas_media_4sem` queda vacía y la predicción es menos confiable. Hay que vigilar especialmente esas estaciones.
- **Factores externos:** el modelo no usa clima, eventos ni cierres de estaciones, así que no anticipa cambios de demanda por esas causas.
- **Cambios de patrón:** la movilidad cambia con la temporada. El monitoreo con Evidently mostró drift solo en `mes`, por diferencia de calendario, y drift leve en `salidas_media_24h` al comparar los mismos meses ([`docs/EVIDENTLY_DATA_DRIFT_P1.md`](docs/EVIDENTLY_DATA_DRIFT_P1.md)). Se recomienda reentrenar con datos recientes y repetir el A/B.
- **Numeración de versiones:** en un despliegue nuevo con Docker Compose, el bootstrap registra el champion como versión 1. La identidad del modelo se sigue con los tags `origen_version` y `origen_run_id`, no con el número de versión local.

## Trazabilidad y gobernanza

- Cada run registra `git_commit`, `data_version`, parámetros, métricas y el modelo con firma y ejemplo de entrada.
- Promover o revertir es mover el alias `champion`; la versión 1 conserva `champion_anterior` para volver atrás.
- El pipeline de CI/CD verifica en cada push a `main` que el champion servido por Compose tenga su linaje (`origen_run_id`, `git_commit`, `data_version`).

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220–229.

Se utilizó Claude (Anthropic) como herramienta de apoyo en la elaboración de esta actividad.
