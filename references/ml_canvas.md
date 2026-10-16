# ML Canvas — Proyecto MiBici (Equipo 42)

TC5061 MLOps · Etapa 1 · Versión 2 (16 oct 2026). P2 queda alineada a la definición de Víctor (PR #5): clasificación de 3 clases por estación y hora. Lo de P3 sigue como propuesta para que Andrea la ajuste con sus resultados. La versión visual de este Canvas está en `references/ml_canvas.png`.

Formato basado en el Machine Learning Canvas de Louis Dorard.

## Resumen de las preguntas

| | P1 | P2 | P3 |
|---|---|---|---|
| Tarea | Regresión | Clasificación multiclase (3 clases) | Clustering |
| Qué se predice o agrupa | Viajes que inician en la estación | Nivel de riesgo de la estación: bajo, desabasto (se queda sin bicis) o saturación (se queda sin anclajes) | Grupos de estaciones con perfil de uso parecido |
| Granularidad | Estación × hora | Estación × hora | Estación |
| Horizonte | Siguiente hora | Siguiente hora | No predictivo; se recalcula cada trimestre |
| Usuario | Operador del sistema | Equipo de rebalanceo | Planeación urbana y expansión |

---

## 1. Propuesta de valor
- **Operador (P1):** anticipar la demanda de la siguiente hora en cada estación para planear la flota en lugar de reaccionar a quejas.
- **Rebalanceo (P2):** saber, una hora antes, qué estaciones tienen riesgo de quedarse sin bicis y cuáles de quedarse sin anclajes, para llevar o retirar bicis antes de que el usuario se quede sin servicio.
- **Planeación (P3):** identificar tipos de estaciones (por ejemplo, de origen matutino, de destino laboral o de uso recreativo de fin de semana) y ver dónde se ubican para decidir expansión y capacidad.

## 2. Tarea de ML
- P1: regresión de un conteo (viajes por estación y hora).
- P2: clasificación multiclase (0 bajo riesgo, 1 desabasto, 2 saturación) con una etiqueta construida a partir del flujo neto de la hora y la capacidad, porque los datos no incluyen el inventario de bicis.
- P3: clustering no supervisado sobre el perfil de uso de cada estación.

## 3. Decisiones
- P1: cuántas bicis conviene tener en cada estación o zona en la siguiente hora.
- P2: a qué estaciones llevar bicis (riesgo de desabasto) y de cuáles retirarlas (riesgo de saturación) en la siguiente hora, y en qué orden.
- P3: en qué zonas abrir o ampliar estaciones y qué capacidad darles según el tipo de uso.

## 4. Cómo se hacen las predicciones
- Etapa 1: solo evaluación offline sobre validación y test.
- Etapas siguientes: P1 y P2 en batch al cierre de cada hora (servicio con FastAPI + Docker); P3 recalculado cada trimestre.

## 5. Evaluación offline
- Partición temporal: entrenamiento jul-2025 → mar-2026 (9 meses), validación abr → jun-2026 (3 meses) y test jul → sep-2026 (3 meses). Validación y test siempre son posteriores al entrenamiento.
- Los hiperparámetros se eligen con validación; el modelo final se reentrena con entrenamiento + validación (12 meses) y se reporta una sola vez en test.
- Cada modelo se compara contra su baseline con la misma partición.
- En P2 las clases de riesgo son pocas (cerca del 1.7 % del test), así que accuracy no sirve para comparar modelos: un modelo que siempre dice "bajo riesgo" tendría accuracy alta y no avisaría de nada. Por eso se evalúa con macro-F1 y con recall y precision de cada clase de riesgo.

## 6. Fuentes de datos
- Viajes mensuales del portal de datos abiertos de MiBici, jul-2025 a sep-2026: 15 archivos `datos_abiertos_AAAA_MM.csv` (enlaces en `config.py`), **5,616,821 viajes** después de la limpieza.
- Nomenclatura de estaciones `nomenclatura_2026_09.csv`: **484 estaciones** con id, nombre, latitud, longitud y capacidad; 4 tienen capacidad 0.
- El sistema crece dentro del rango: de 368,186 viajes y 367 estaciones con viajes en jul-2025 a 490,056 viajes y 468 estaciones en sep-2026.
- No se usan datos externos en esta etapa (clima y días festivos quedan como mejora).

## 7. Recolección y versionado de datos
- `dataset.py` descarga los meses y la nomenclatura, normaliza columnas, codificación y fechas, quita duplicados y guarda Parquet en `data/processed/`. Solo lee las columnas que usan los modelos.
- `manifest.py` genera `data_manifest.json` (versionado en Git) con número de registros y hash SHA-256 de cada archivo; `make data` corre los dos pasos. MLflow registra como `data_version` el hash del manifiesto.
- Las columnas de usuario (`usuario_id`, `genero`, `anio_nacimiento`) no se leen, no se guardan en los datos procesados y no se registran en MLflow.

## 8. Variables
Todas se calculan con información disponible al cierre de la hora anterior a la que se predice (rezagos con `shift` sobre la cuadrícula completa de horas y promedios que solo miran hacia atrás).
- **Base común:** tabla estación × hora con salidas, llegadas, flujo neto (llegadas − salidas) y capacidad; las horas sin viajes van en 0 y cada estación entra desde su primer viaje.
- **P1:** hora del día, día de la semana, fin de semana, mes; viajes de la estación hace 1, 2, 24 y 168 horas; promedio de las últimas 24 horas; promedio de la misma hora en las 4 semanas anteriores; capacidad.
- **P2 (implementadas en el PR #5):** capacidad; hora del día, día de la semana, fin de semana y mes; salidas de la estación hace 1, 2, 24 y 168 horas; promedio de salidas de las últimas 24 horas y de la misma hora en las 4 semanas anteriores. Se excluyen las estaciones con capacidad 0 (no se puede calcular el flujo relativo). Mejora prevista: agregar los mismos rezagos para llegadas y flujo neto, que deberían ayudar sobre todo a detectar saturación.
- **P3:** proporción de salidas y de llegadas por hora (24 h) en días entre semana y en fin de semana, y flujo neto promedio relativo a la capacidad. La ubicación no entra al clustering; se usa para el mapa.

## 9. Construcción de modelos
- Un baseline sencillo y al menos un candidato por pregunta (detalle en la tabla de modelos y métricas).
- Todo se entrena desde el código del repositorio y se registra en MLflow (experimento `mibici-equipo42`) con parámetros, métricas, etiquetas y artefactos, mediante la función común `registrar_run`.
- Reentrenamiento previsto: mensual, al publicarse un mes nuevo, o al detectar drift.

## 10. Monitoreo (etapas siguientes)
- Drift de las variables de entrada con Evidently. Ya se observa un cambio entre jul-2025 y sep-2026 (+33 % de viajes y 101 estaciones más con viajes), así que el monitoreo es necesario desde el inicio.
- Seguimiento de `test_mae` (P1) y de `test_macro_f1` y el recall de cada clase de riesgo (P2) contra el baseline en cada mes nuevo; si el modelo deja de superarlo, se reentrena.

---

## Formulación detallada

### P1 — Viajes por estación en la siguiente hora
- **Qué se predice:** al cierre de cada hora *t*, el número de viajes que iniciarán en la estación *s* durante la hora siguiente [*t*, *t* + 1 h).
- **Granularidad y horizonte:** estación × hora; siguiente hora.
- **Relevancia:** la demanda cambia mucho por hora y por estación (entre semana hay picos a las 8 h y de 17 a 19 h; en fin de semana el uso se concentra a mediodía). Saberlo con una hora de anticipación permite mover bicis antes de que falten.
- **Quién y qué decide:** el operador decide dónde concentrar la flota y qué estaciones priorizar en la siguiente hora.

### P2 — Riesgo de quedarse sin bicis o sin anclajes
- **Qué se predice:** al cierre de cada hora, la clase de riesgo de la estación *s* (capacidad *C*) en la hora siguiente *h*, usando solo información de horas anteriores.
- **Etiqueta** (`crear_etiqueta_p2` en `features.py`): flujo relativo *r* = (llegadas − salidas de la hora *h*) / *C*.
  - Clase 1, desabasto: *r* < −0.30 (en una hora salen, en neto, más del 30 % de la capacidad).
  - Clase 2, saturación: *r* > +0.30 (en una hora llegan, en neto, más del 30 % de la capacidad).
  - Clase 0, bajo riesgo: el resto.
- **Granularidad y horizonte:** estación × hora; siguiente hora.
- **Justificación de la aproximación** *(redactada por el equipo a partir del README y el código del PR #5; pendiente de validar por Víctor)*:
  - Sin inventario, el flujo neto relativo a la capacidad es la mejor señal disponible de presión sobre la estación: perder o ganar en neto casi un tercio de los anclajes en una sola hora deja poco margen, aunque la estación empiece a media capacidad.
  - Se usa una hora para que P2 tenga la misma granularidad y horizonte que P1: comparte la tabla estación × hora, las variables de `features.py` y la partición, y el resultado llega a tiempo para que el equipo de rebalanceo actúe en la hora siguiente.
  - El umbral es simétrico porque los dos casos dejan al usuario sin servicio: sin bici para salir o sin anclaje para dejarla. Por la misma razón las dos clases de riesgo tienen el mismo peso al entrenar.
  - Con este umbral las clases de riesgo son poco frecuentes (7,654 de desabasto y 8,651 de saturación en 961,886 estación-horas del test, cerca del 1.7 %), lo que corresponde a eventos fuera de lo normal y no a la operación de todos los días.
- **Alternativa considerada:** una etiqueta binaria con ventana de 3 horas (riesgo si el flujo neto acumulado baja de −0.3·*C* o sube de 0.6·*C* en algún momento de las 3 horas siguientes). En jul-2025 dejaba 7.34 % de positivos (6.17 % vaciarse, 1.18 % llenarse). Se descartó a favor de la definición de 3 clases porque separa desabasto de saturación, que piden acciones distintas (llevar o retirar bicis), y mantiene el mismo horizonte que P1. A cambio, la ventana de una hora no capta acumulaciones que se forman en varias horas seguidas; queda como mejora.
- **Limitaciones:** se desconoce el inventario inicial y el rebalanceo con camionetas no aparece en los datos, así que la etiqueta mide presión de demanda, no disponibilidad real. La capacidad viene de la nomenclatura de sep-2026 y se usa para todo el periodo.
- **Quién y qué decide:** el equipo de rebalanceo decide a qué estaciones llevar bicis, de cuáles retirarlas y en qué orden.

### P3 — Grupos de estaciones por patrón de uso
- **Qué se agrupa:** cada estación con historia suficiente en el periodo de entrenamiento, según su perfil de uso.
- **Granularidad y horizonte:** estación; descriptivo, se recalcula cada trimestre. El periodo de test se usa para revisar que los grupos sean estables.
- **Relevancia:** muestra qué zonas generan viajes en la mañana, cuáles los reciben y cuáles se usan sobre todo en fin de semana; eso orienta dónde crecer y con qué capacidad.
- **Quién y qué decide:** planeación urbana decide zonas de expansión y capacidad de estaciones nuevas, comparándolas con grupos de estaciones parecidas.

---

## Modelos y métricas

| | Baseline | Candidato(s) | Métricas | Por qué esas métricas |
|---|---|---|---|---|
| **P1** | Naive estacional: viajes de la misma estación a la misma hora una semana antes | `HistGradientBoostingRegressor` con `loss="poisson"`; `Ridge` como alternativa lineal | **`test_mae`**, `test_rmse` (y sus versiones `val_`) | MAE queda en viajes por hora y se explica fácil al operador; RMSE castiga los errores grandes, que se concentran en las horas pico. MAPE se descarta porque el 53.1 % de las estación-horas tiene 0 salidas. La pérdida Poisson se usa al entrenar el candidato porque el objetivo es un conteo. |
| **P2** | Naive estacional: la clase de la misma estación a la misma hora una semana antes (rezago de 168 h de la etiqueta) | `RandomForestClassifier` (50 árboles, `max_depth=12`, `min_samples_leaf=20`, `class_weight={0: 1, 1: 10, 2: 10}`) | **`test_macro_f1`**, **`test_recall_desabasto`**, **`test_recall_saturacion`**, `test_precision_desabasto`, `test_precision_saturacion`; `test_accuracy` solo como referencia | Con cerca de 1.7 % de clases de riesgo, accuracy engaña. Macro-F1 promedia las tres clases con el mismo peso, así que la clase mayoritaria no tapa a las de riesgo. El recall de cada clase de riesgo mide cuántos eventos se detectan (no avisar es lo más caro, porque el usuario se queda sin servicio) y la precision cuántas alertas son reales (cada falsa alarma es un viaje de camioneta de más). Los pesos de clase compensan el desbalance para que el modelo no prediga siempre "bajo riesgo". |
| **P3** | `KMeans` con *k* elegido por silhouette (*k* = 3…8) | PCA + `KMeans`; `AgglomerativeClustering` (Ward) | **`train_silhouette`**, `train_davies_bouldin`, `test_estabilidad_ari` | Sin etiquetas, se usan métricas internas de separación y compactación. La estabilidad (ARI entre los grupos de entrenamiento y la asignación con perfiles del test) revisa que los grupos no sean ruido de un solo periodo. |

**Resultado del baseline de P1** (run `p1-baseline-semana-anterior`): `val_mae` 0.993, `val_rmse` 2.004, `test_mae` 0.942, `test_rmse` 1.921 viajes por estación-hora. El error se concentra en las horas pico (cerca de 1.8 viajes a las 18 h y 1.4 a las 8 h) y es casi 0 de 1 a 4 h.

**Resultado preliminar del candidato de P2** (de `reports/metrics/metricas_p2.json` en el PR #5; todavía no es un run de MLflow, las cifras finales salen del run): en test, macro-F1 0.46; desabasto con recall 0.30 y precision 0.17; saturación con recall 0.19 y precision 0.20; accuracy 0.97. La accuracy alta con recall bajo en las clases de riesgo confirma que accuracy no sirve como métrica principal en P2. El baseline naive de P2 está pendiente de implementarse y registrarse.

---

## Decisiones y justificación

| Decisión | Justificación |
|---|---|
| Rango de 15 meses (jul-2025 → sep-2026), no 18 ni 24 | La estacionalidad anual es débil (con 24 meses, unos 382 mil viajes al mes estables) y lo que domina es la hora y el día de la semana. Además el sistema creció mucho dentro del propio rango (de 367 a 468 estaciones con viajes), así que los datos más viejos representan peor el servicio actual. Con 15 meses el modelo final se entrena con 12 y cada mes del test tiene su equivalente del año anterior. |
| Partición 9 / 3 / 3 por fecha | Hace falta un periodo de validación para elegir hiperparámetros sin tocar el test. Al ser temporal, el modelo nunca ve información posterior a lo que evalúa. |
| P1 por hora y no por día | El operador decide hora a hora. Los muchos ceros se manejan con MAE/RMSE en la evaluación y con pérdida Poisson en el candidato. |
| P2 con 3 clases en la siguiente hora y umbral ±0.30 (definición de Víctor) | Separa desabasto de saturación, que piden acciones distintas, y comparte granularidad, horizonte y variables con P1. Se prefirió sobre la etiqueta binaria con ventana de 3 h, que queda como alternativa considerada (ver formulación de P2). |
| Baseline de P2 = clase de hace una semana | Es la misma lógica que el baseline de P1: captura el patrón semanal sin entrenar nada, así que cualquier modelo tiene que superarlo para justificar su complejidad. |
| Cada estación entra desde su primer viaje | Entre jul-2025 y sep-2026 aparecieron 101 estaciones con viajes; llenar con ceros el periodo antes de que existieran sesgaría rezagos y métricas. |
| Sin datos personales en datos procesados ni en MLflow | Ninguna pregunta usa `usuario_id`, `genero` ni `anio_nacimiento`; no leerlos aplica minimización de datos y cumple la buena práctica de no registrar datos personales. `registrar_run` rechaza parámetros con esos nombres. |
