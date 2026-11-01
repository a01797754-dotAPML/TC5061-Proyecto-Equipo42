$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "============================================================"

    & $Command

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: fallo el paso '$Name'." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host "OK: $Name" -ForegroundColor Green
}

Write-Host ""
Write-Host "TC5061 - Equipo 42 | Actividad 7"
Write-Host "Pruebas, DeepChecks y chequeos de seguridad"
Write-Host ""

Invoke-Step "1/5 - PyTest en entorno mibici-mlops" {
    conda run --no-capture-output -n mibici-mlops `
        python -m pytest tests
}

Invoke-Step "2/5 - Preparar tabla P1 requerida por DeepChecks" {
    conda run --no-capture-output -n mibici-mlops `
        python -c "from mibici_mlops.modeling.p1 import construir_tabla; t=construir_tabla(); print('P1 tabla:', t.shape)"
}

Invoke-Step "3/5 - Validacion de datos con DeepChecks" {
    conda run --no-capture-output -n mibici-dc `
        python -m mibici_mlops.validacion

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    conda run --no-capture-output -n mibici-dc `
        python -m pytest tests/test_validacion.py
}

Invoke-Step "4/5 - Bandit" {
    conda run --no-capture-output -n auditoria `
        python -m bandit -r mibici_mlops -ll -q
}

Invoke-Step "5/5 - Auditoria de dependencias con pip-audit" {
    conda run --no-capture-output -n auditoria `
        python -m pip_audit -r requirements.txt

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    conda run --no-capture-output -n auditoria `
        python -m pip_audit -r requirements-deepchecks.txt `
        --ignore-vuln PYSEC-2026-3447 `
        --ignore-vuln GHSA-xmj3-9gf7-h52w
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "ACTIVIDAD 7: TODOS LOS CHEQUEOS TERMINARON CORRECTAMENTE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
