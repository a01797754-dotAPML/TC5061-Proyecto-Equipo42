# Pipeline de CI/CD con GitHub Actions — Actividad 10

El workflow está en [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Se ejecuta solo en cada `push` a `main` y en cada Pull Request hacia `main`. También se puede lanzar a mano desde la pestaña **Actions → CI/CD → Run workflow**.

## Qué hace

```
push / pull request
        │
        ▼
 ┌──────────────────────────┐
 │ tests (matriz)            │  ubuntu-latest y windows-latest, Python 3.12
 │ pip install + pytest      │  caché de pip
 └────────────┬─────────────┘
              │ si pasan
     ┌────────┴─────────┐
     ▼                  ▼
┌─────────────┐   ┌──────────────────────────────┐
│ build       │   │ compose                       │
│ imagen de la│   │ docker compose up -d --build  │
│ API (Buildx)│   │ /health, /predict (200), 422  │
│ push a GHCR │   │ docker stats, down -v         │
│ solo en main│   └──────────────────────────────┘
└─────────────┘
     └────────┬─────────┘
              ▼
      resumen (tabla de resultados)
```

| Job | Qué valida | Por qué |
|---|---|---|
| `tests` | Todas las pruebas de `tests/` en Ubuntu y Windows | El equipo trabaja en Windows y macOS, y el contenedor corre en Linux. La matriz revisa que el código funcione en los dos sistemas y `fail-fast: false` deja ver el resultado de ambos aunque uno falle. |
| `build` | Que la imagen se construya con el `Containerfile` | Si falla la imagen, no hay despliegue. En `main` la imagen se publica en GitHub Container Registry con dos etiquetas: el SHA corto del commit (trazabilidad) y `latest`. En los PRs solo se construye, no se publica. |
| `compose` | Que el sistema completo se levante con un solo comando y responda | Es el despliegue en un entorno limpio: levanta MLflow, registra el champion (bootstrap) y la API, y hace un smoke test real de `/health`, `/predict` y de una entrada inválida (422). Al final guarda el consumo de CPU y memoria de los contenedores. |
| `resumen` | Tabla con el resultado de cada job | Queda en el resumen de la ejecución, como notificación visible para todo el equipo en cada PR. GitHub además manda correo cuando un workflow falla. |

## Optimizaciones del pipeline

- **Caché de dependencias:** `actions/setup-python` guarda la caché de pip con `requirements.txt` como llave, y Buildx guarda las capas de la imagen en la caché de GitHub Actions (`type=gha`). Las corridas siguientes no reinstalan todo desde cero.
- **Jobs en paralelo:** `build` y `compose` corren al mismo tiempo, después de las pruebas.
- **`concurrency` con `cancel-in-progress`:** si se hace otro push a la misma rama, se cancela la corrida anterior en lugar de gastar minutos en un commit viejo.
- **`paths-ignore`:** cambios solo de documentación (`*.md`, `docs/`, `references/`, `notebooks/`) no disparan el pipeline.
- **Permisos mínimos:** el token solo tiene lectura, salvo `packages: write` en el job que publica la imagen.

## Cómo usarlo

1. Crear una rama, hacer cambios y abrir el PR hacia `main`. El pipeline corre solo y sus checks aparecen en el PR.
2. Revisar el resultado en la pestaña **Actions** del repositorio o en los checks del PR. En la página de cada ejecución está el resumen con las pruebas, la imagen y la respuesta de `/predict`.
3. Al hacer merge a `main`, el pipeline corre otra vez y publica la imagen en GHCR. Se puede descargar con:

```powershell
docker pull ghcr.io/a01797754-dotapml/mibici-api:latest
```

Para correr las mismas pruebas en local antes del push (PowerShell, con el entorno `mibici` activo y desde la raíz del repo):

```powershell
python -m pytest tests -q
docker compose config --quiet
```

## Resultados

_Pendiente: se llena con la primera ejecución en verde (duración por job, número de pruebas y respuesta de `/predict`)._
