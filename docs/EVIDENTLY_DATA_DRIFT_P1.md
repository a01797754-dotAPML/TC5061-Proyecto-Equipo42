# Monitoreo de Data Drift y calidad de datos — MiBici P1

## 1. Objetivo

Evaluar cambios en las distribuciones y la completitud de las 11 variables predictoras del modelo champion P1 (`p1-hgb-final`), que estima las salidas de bicicletas por estación y hora.

El monitoreo utiliza Evidently 0.7.23 y corresponde a una evaluación offline sobre datos históricos. Sus resultados no constituyen, por sí solos, evidencia de degradación predictiva.

## 2. Periodos analizados

Se utilizan tres muestras de 50,000 registros y 11 variables, obtenidas mediante muestreo aleatorio reproducible con semilla 42.

| Conjunto | Periodo | Propósito |
|---|---|---|
| Referencia principal | Julio 2025–junio 2026 | Periodo de entrenamiento del champion P1 |
| Referencia estacional | Julio–septiembre 2025 | Comparación de los mismos meses entre años |
| Actual | Julio–septiembre 2026 | Periodo de monitoreo |

Las muestras se almacenan localmente en `reports/evidently/`.

## 3. Metodología

Se utiliza `DataDriftPreset` de Evidently para evaluar cambios en las distribuciones marginales de las variables predictoras y `DataSummaryPreset` para revisar calidad y completitud.

El umbral individual de detección es 0.1 y el criterio global es una proporción de variables con drift de al menos 0.5.

Se realizan tres análisis:

1. Principal: referencia anual contra periodo actual, con las 11 variables.
2. Complementario: misma comparación excluyendo `mes`, únicamente con fines diagnósticos.
3. Estacional: julio–septiembre de 2025 contra julio–septiembre de 2026, con las 11 variables.

Evidently selecciona métodos estadísticos según los datos y tipos inferidos. En estos resultados se utilizan distancia de Wasserstein normalizada y distancia de Jensen-Shannon. Los scores calculados mediante métodos diferentes no deben compararse directamente en magnitud.

La exclusión de `mes` del análisis complementario no modifica las características del modelo champion.

## 4. Resultados de Data Drift

| Indicador | Principal | Sin `mes` | Estacional |
|---|---:|---:|---:|
| Variables evaluadas | 11 | 10 | 11 |
| Variables con drift | 1 | 0 | 1 |
| Proporción con drift | 9.09% | 0% | 9.09% |
| Drift global detectado | No | No | No |

### 4.1. Análisis principal

La única variable con drift es `mes`, con score de 0.730140 (Wasserstein normalizada), superior al umbral de 0.1.

Las otras diez variables no presentan drift bajo la configuración utilizada.

Al excluir `mes` con fines diagnósticos, ninguna de las diez variables restantes presenta drift.

### 4.2. Comparación estacional

Al comparar julio–septiembre de 2025 contra julio–septiembre de 2026:

- `mes` no presenta drift: score de 0.030480 (Jensen-Shannon).
- `salidas_media_24h` presenta drift leve: score de 0.105946 (Wasserstein normalizada), ligeramente superior al umbral de 0.1.
- Las otras nueve variables no superan sus umbrales.

La desaparición del drift de `mes` al comparar los mismos meses respalda una explicación relacionada con las diferencias de calendario entre ventanas.

El drift leve de `salidas_media_24h` sugiere un posible cambio en la distribución de la actividad histórica entre años y amerita seguimiento. No demuestra por sí solo deterioro del modelo.

En ambos análisis se detecta drift en una de once variables, por debajo del criterio global de 50%.

## 5. Resultados de calidad de datos

### 5.1. Completitud de las muestras principales

| Variable | Referencia anual | Actual |
|---|---:|---:|
| `capacidad` | 12 (0.024%) | 188 (0.376%) |
| `salidas_media_4sem` | 3,216 (6.432%) | 1,836 (3.672%) |
| Total de valores faltantes | 3,228 | 2,024 |

Las demás variables no presentan valores faltantes en las muestras analizadas.

Con la referencia anual, el porcentaje de faltantes de `salidas_media_4sem` es menor en el periodo actual, aunque `capacidad` presenta un incremento.

### 5.2. Investigación de `salidas_media_4sem`

Se revisó la tabla completa del periodo julio–septiembre de 2026, no solamente la muestra de 50,000 registros.

Resultados:

- 37,050 registros presentan `salidas_media_4sem` faltante.
- Los faltantes corresponden a 85 estaciones distintas.
- 29,018 registros (78.3%) corresponden a 58 estaciones cuya primera aparición observada en los datos ocurrió a partir de julio de 2026.
- Los 37,050 faltantes (100%) se encuentran dentro de los primeros 28 días desde la primera aparición observada de cada estación.
- No se encontraron faltantes posteriores a esos primeros 28 días.

La implementación en `mibici_mlops/features.py` calcula la característica mediante cuatro desplazamientos históricos:

`sum(g.shift(168 * k) for k in range(1, 5)) / 4`

Por ello, el cálculo requiere cuatro observaciones históricas desplazadas, incluida la de 672 posiciones anteriores. Si falta alguna, el resultado puede ser nulo.

Estos hallazgos respaldan que la falta de historial suficiente explica los valores faltantes iniciales. Los desplazamientos se expresan en observaciones; su equivalencia con semanas supone registros horarios continuos por estación. La primera aparición en el conjunto de datos no necesariamente coincide con la fecha oficial de apertura de la estación.

## 6. Limitaciones

- El monitoreo utiliza muestras aleatorias de 50,000 registros por conjunto.
- Las pruebas detectan diferencias en distribuciones marginales, no degradación predictiva.
- Los resultados dependen de los métodos estadísticos, tipos inferidos y umbrales utilizados.
- Los scores de Wasserstein y Jensen-Shannon no son directamente comparables.
- El análisis sin `mes` es diagnóstico y no cambia el modelo original.
- La comparación estacional controla los meses observados, pero no elimina otras diferencias entre años.
- La asociación de faltantes con el historial inicial no acredita fechas oficiales de apertura de estaciones.
- El monitoreo no justifica automáticamente un reentrenamiento; debe complementarse con métricas predictivas y pruebas A/B.

## 7. Reproducción

El análisis se ejecuta offline desde la raíz del repositorio con Python 3.12.

### 7.1. Preparar las muestras

Se requiere acceso a los datos fuente de MiBici y al entorno principal del proyecto.

Ejecutar:

    .venv/bin/python scripts/preparar_muestras_evidently.py

El script utiliza `construir_tabla()` del modelo P1 y genera tres muestras reproducibles de 50,000 registros, con semilla 42:

- `referencia_p1.parquet`: julio 2025–junio 2026.
- `referencia_estacional_p1.parquet`: julio–septiembre 2025.
- `actual_p1.parquet`: julio–septiembre 2026.

### 7.2. Instalar Evidently

Crear un entorno independiente con Python 3.12:

    python3.12 -m venv .venv-evidently

Instalar dependencias:

    .venv-evidently/bin/python -m pip install -r requirements-evidently.txt

### 7.3. Ejecutar el monitoreo

Ejecutar:

    .venv-evidently/bin/python scripts/evidently_drift_p1.py

El script genera los reportes HTML y archivos CSV de drift y calidad de datos en `reports/evidently/`.

### 7.4. Control de versiones

El directorio `reports/evidently/` está excluido mediante `.gitignore`. Los archivos Parquet, HTML y CSV permanecen locales y pueden regenerarse mediante los scripts versionados.

## 8. Evidencias

- `reports/evidently/reporte_drift_p1.html`
- `reports/evidently/reporte_drift_p1_sin_mes.html`
- `reports/evidently/reporte_drift_p1_estacional.html`
- `reports/evidently/resultados_drift_p1.csv`
- `reports/evidently/resultados_drift_p1_estacional.csv`
- `reports/evidently/resultados_calidad_p1.csv`

## 9. Conclusión

No se detectó drift global bajo los umbrales establecidos. El análisis principal identifica drift únicamente en `mes`, mientras que la comparación estacional elimina esa señal y detecta un drift leve en `salidas_media_24h`.

La investigación de calidad de datos muestra que todos los faltantes observados en `salidas_media_4sem` durante julio–septiembre de 2026 se concentran en los primeros 28 días de historial observado de las estaciones, lo cual es consistente con su fórmula de construcción.

Se recomienda monitorear `salidas_media_24h` y complementar estos hallazgos con las métricas predictivas y las pruebas offline A/B del equipo antes de decidir sobre un eventual reentrenamiento.
