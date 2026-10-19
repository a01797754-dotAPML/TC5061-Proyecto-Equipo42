# Plan de trabajo · Entrega 2 (MLflow Project y tracking avanzado)

**Equipo 42 · TC5061** · Entrega: **domingo 25 de octubre, 23:59** · Reporte: `TC5061_Reporte_MLFlow_Equipo42.pdf`

## Qué tenemos que lograr

1. **Proyecto ejecutable con un solo comando** (`mlflow run`): ingesta y entrenamiento de P1, P2 y P3.
2. **Al menos 3 modelos candidatos por pregunta**, además del baseline de la entrega 1, comparables en el mismo experimento (`mibici-equipo42`), con **al menos una búsqueda de hiperparámetros** organizada como nested runs.
3. **Evidencia completa en cada candidato:** modelo, versión de los datos, gráficas y configuración.
4. **Un modelo en producción por pregunta** en el Model Registry.
5. **Reporte** con tabla comparativa, justificación del modelo elegido y capturas.
6. **Cada integrante: al menos 1 Pull Request propio y 1 revisión** en esta entrega (se evalúa individualmente).

## Quién hace qué

| Integrante | Responsable de | Entrega concreta |
|---|---|---|
| **Carlos C.** | Integración y piezas compartidas | Estructura del MLflow Project (comandos `ingest` y `train`), registro común de evidencias y etiquetas, README con los comandos. Corrida oficial, Model Registry (los 3 modelos en producción) y capturas. Integración del reporte final |
| **Carlos E.** | **P1** · demanda por estación | 3+ candidatos (por ejemplo HistGradientBoosting, Ridge y una búsqueda de hiperparámetros) con sus gráficas. Tabla comparativa y justificación del modelo elegido de P1 |
| **Víctor** | **P2** · riesgo de desabasto o saturación | 3+ candidatos (por ejemplo Random Forest ajustado, una variante con nuevas variables y una búsqueda de hiperparámetros) con sus gráficas, incluida la curva precisión-recall. Tabla comparativa y justificación del modelo elegido de P2 |
| **Andrea** | **P3** · grupos de estaciones | 3+ candidatos (por ejemplo KMeans sin escalar las proporciones, PCA + flujo neto y una búsqueda de *k*) con mapa o perfiles de grupos. Tabla comparativa y justificación del modelo elegido de P3 |

**Revisiones cruzadas:** Carlos C. revisa los PRs de P1, P2 y P3 (porque los integra). Los PRs de Carlos C. los revisan al menos dos personas del equipo. Además, cada quien revisa el PR de otra pregunta.

## Calendario

| Día | Qué pasa | Quién |
|---|---|---|
| **Lun 19** | Acordar este plan en el chat | Todos |
| **Mar 20 – Mié 21 (mediodía)** | PR de las piezas compartidas (MLflow Project + registro común). **Hay que revisarlo y hacer merge el miércoles a mediodía**, porque desbloquea a todos | Carlos C. + revisores |
| **Mar 20 – Mié 21** | Mientras tanto: preparar los candidatos de su pregunta en su rama (el código del modelo no depende del PR compartido) | Carlos E., Víctor, Andrea |
| **Mié 21 – Vie 23 (18:00)** | PR de cada pregunta con sus 3+ candidatos y su búsqueda, **probado localmente con `mlflow run`**. Merge a más tardar el **viernes a las 18:00** | Carlos E., Víctor, Andrea |
| **Vie 23 (noche)** | Ensayo completo desde un clon limpio, para detectar problemas antes del sábado | Carlos C. |
| **Sáb 24 (día)** | Cada quien redacta en el Google Doc su sección: tabla comparativa, análisis y justificación de su pregunta, con sus resultados locales. Revisiones pendientes de PRs | Carlos E., Víctor, Andrea |
| **Sáb 24 (noche)** | Corrida oficial, Model Registry y capturas | Carlos C. |
| **Dom 25** | Actualizar cifras del reporte con la corrida oficial, README final, PDF. **Subir a Canvas antes de las 20:00** (margen de 4 horas) | Carlos C. + todos revisan |

## Cómo compartimos resultados (una sola instancia de MLflow)

- Cada quien **desarrolla y prueba en su MLflow local**: ahí explora, compara y escribe su análisis.
- Los **resultados oficiales** (los que van al reporte, al Model Registry y a las capturas) se generan **una sola vez**, en una sola instancia, corriendo los mismos comandos `mlflow run` desde un clon limpio.
- Funciona porque todo es reproducible: mismo código (commit) y mismos datos (`data_version`) dan las mismas cifras. Ya lo comprobamos en la entrega 1.
- Por eso: **lo que no entre a `main` por PR no estará en la corrida oficial.**

## Acuerdos para no repetir los problemas de la entrega 1

1. **Una pregunta, un dueño:** nadie modifica el código de otra pregunta sin avisar. Si algo de lo compartido no te sirve, coméntalo en el chat o en el PR de Carlos C.; no lo dupliques.
2. **Mismas reglas de comparación:** la misma partición de datos y las mismas métricas de la entrega 1. Si alguien necesita cambiar algo, se justifica en el reporte.
3. **Nombres claros** para las corridas y las etiquetas que propone el registro común, para poder filtrar en MLflow.
4. **PRs pequeños y a tiempo:** mejor dos PRs revisables que uno enorme el sábado.
5. **Antes de `git add`, un `git status`:** nada de datos, `mlflow.db`, `mlruns/` ni modelos en el repo.
6. **Avisar en el chat** cuando un PR esté listo para revisión, y **revisar dentro de las 24 horas siguientes**.

## Definición de "terminado" para cada pregunta

- [ ] PR con merge en `main`
- [ ] Se ejecuta con `mlflow run` (sin notebooks)
- [ ] 3+ candidatos además del baseline, y 1 búsqueda con nested runs
- [ ] Cada candidato guarda: modelo, versión de datos, gráficas y configuración
- [ ] Sección del reporte escrita: tabla comparativa contra el baseline y justificación del modelo elegido (mejora, métrica de negocio, estabilidad y costo)

---

## Anexo técnico: contratos para implementar (para quien programe, y para su asistente de IA)

> Fuente de verdad de las convenciones de la entrega 2. Si algo aquí no encaja con tu pregunta, coméntalo en el chat o en un PR antes de implementar otra variante. Implementado en el PR #14 (piezas compartidas); el ejemplo completo de uso está en `tests/test_mlflow_project.py`.

**Base que no cambia (entrega 1):** misma partición (`VAL_INICIO`, `TEST_INICIO` en `config.py`), mismas métricas y nombres:
- P1: `val_mae`, `test_mae` (principal), `val_rmse`, `test_rmse`
- P2: `test_macro_f1`, `test_recall_desabasto`, `test_recall_saturacion` (principales), `test_precision_desabasto`, `test_precision_saturacion`, `test_accuracy` (referencia), y sus versiones `val_`
- P3: `train_silhouette`, `train_davies_bouldin`, `test_estabilidad_ari`

Rutas, fechas y semilla siempre desde `config.py`; nunca escritas a mano.

**Ejecución:** todo con el MLflow Project, desde la raíz del repo (con el servidor levantado y `MLFLOW_TRACKING_URI` exportada, ver abajo):
```bash
python -m mlflow run . -e ingest --experiment-name mibici-equipo42
python -m mlflow run . -e train --experiment-name mibici-equipo42 \
    -P question=p1 -P model=<modelo> -P params='{"<hiperparametro>": <valor>}'
```
- `--experiment-name mibici-equipo42` es obligatorio: sin él, `mlflow run` registra en el experimento *Default*.
- `model` es obligatorio (cada pregunta define sus modelos); `params` es un JSON con los hiperparámetros (por defecto `{}`).
- La primera ejecución tarda unos minutos porque pyenv crea el entorno de `python_env.yaml`; después se reutiliza.
- Prueba rápida sin MLflow Project: `python -m mibici_mlops.modeling.train --question p1 --model <modelo> --params '{...}'`.

**Interfaz por pregunta:** `mibici_mlops/modeling/pN.py` expone
```python
def train(model: str, params: dict) -> str:  # devuelve el run_id
```
El punto de entrada `train` la llama según `question`. Las funciones `main()` de la entrega 1 (baselines) se conservan.

**Registro en MLflow:** con el helper común de `mibici_mlops/tracking.py` (no llamar `mlflow.log_*` a mano para lo común):
```python
from mibici_mlops.tracking import iniciar_run, registrar_resultados

def train(model: str, params: dict) -> str:
    ...  # construir tablas y X_train, y_train con features.py
    with iniciar_run("P1", f"p1-{model}", model_family="hist_gradient_boosting",
                     stage_of_work="candidate", params={"modelo": model, **params}) as run:
        estimador.fit(X_train, y_train)
        ...  # métricas val_/test_ en las mismas filas que el baseline, gráficas en una carpeta temporal
        registrar_resultados(metricas, modelo=estimador, X_ejemplo=X_train, graficas=[...],
                             config={...}, datos_entrenamiento=tabla_train_agregada,
                             nombre_datos="p1-estacion-hora-train")
    return run.info.run_id
```
`iniciar_run`:
- reutiliza el run que crea `mlflow run` (no abre otro) y le pone el nombre descriptivo;
- agrega tags obligatorios: `equipo`, `pregunta`/`question` (`P1`/`P2`/`P3`), `git_commit`, `git_cambios_sin_commit`, `model_family` (por ejemplo `hgb`, `rf`, `kmeans`) y `stage_of_work` (`baseline`, `search`, `candidate`);
- agrega params comunes: fechas, `val_inicio`, `test_inicio`, `data_version`, `seed`;
- valida `stage_of_work` y rechaza parámetros con datos personales.

`registrar_resultados`:
- registra las métricas (rechaza las que no tengan prefijo `train_`, `val_` o `test_`);
- organiza artefactos en subcarpetas: `plots/` (gráficas), `config/run_config.json` (configuración) y `data/data_manifest.json` (versión de datos);
- registra el conjunto de entrenamiento con `mlflow.log_input` (digest = `data_version`);
- registra el modelo con `signature` e `input_example`. **En MLflow 3 el modelo es un *Logged Model***: aparece en la sección *Logged models* del run (URI `models:/m-...`, guardado en el tag `model_uri`), no como carpeta en *Artifacts*. Ese URI es el que se registra en el Model Registry;
- MLflow 3 guarda los modelos de scikit-learn con **skops** (no ejecuta código al cargar). Si al guardar sale `UntrustedTypesFoundException`, declarar **solo** el tipo concreto en `tipos_confiables=[...]` (por ejemplo, HistGradientBoosting necesita `"sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"`). Devuelve el `model_uri`.

**Nombres de runs:** `pN-<modelo>-<variante>`, por ejemplo `p1-hgb-poisson`, `p2-rf-hora-t`, `p3-kmeans-sin-escalar`. Búsquedas: run padre `pN-<modelo>-search` (`stage_of_work=search`) con runs hijos `iniciar_run(..., nested=True)`, uno por combinación (ver `test_busqueda_con_nested_runs`). La selección se hace con métricas de **validación**; test solo para el reporte.

**Gráficas mínimas (en `plots/`):** P1 real vs. predicho y residuales; P2 matriz de confusión y curva precisión-recall; P3 mapa y perfiles de los grupos.

**Datos:** registrar con `mlflow.log_input` solo tablas **agregadas** (estación × hora o perfil por estación), nunca viajes individuales ni columnas de usuario.

**Model Registry (lo hace Carlos C. en la corrida oficial):** `mibici-equipo42-p1-demanda`, `mibici-equipo42-p2-riesgo`, `mibici-equipo42-p3-grupos`; alias `champion` para producción y, opcional, `challenger`. Carga: `models:/mibici-equipo42-p1-demanda@champion`.

**Servidor de tracking local:**
```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```
Usar `python -m mlflow` (no el `mlflow` del PATH, que puede ser de otra versión). Nunca correr `mlflow db upgrade`.

**Entorno:** `python_env.yaml` con las mismas versiones de `requirements.txt`; requiere `pyenv` (`brew install pyenv && pyenv install 3.12.15`).

**Git:** rama `tipo/descripcion` desde `main` actualizado, Conventional Commits, `git status` antes de `git add`, PR con al menos una revisión.
