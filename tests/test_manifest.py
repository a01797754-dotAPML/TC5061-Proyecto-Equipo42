"""Pruebas unitarias del manifiesto SHA-256 de los datos."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from mibici_mlops import manifest


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "PROJ_ROOT", tmp_path)
    return tmp_path


def test_sha256_coincide_con_hashlib(tmp_path):
    archivo = tmp_path / "datos.csv"
    archivo.write_bytes(b"a,b\n1,2\n")
    assert manifest.sha256(archivo) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_sha256_de_archivo_vacio(tmp_path):
    archivo = tmp_path / "vacio.csv"
    archivo.touch()
    assert manifest.sha256(archivo) == hashlib.sha256(b"").hexdigest()


def test_sha256_cambia_si_cambia_el_contenido(tmp_path):
    archivo = tmp_path / "datos.csv"
    archivo.write_text("1")
    antes = manifest.sha256(archivo)
    archivo.write_text("2")
    assert manifest.sha256(archivo) != antes


def test_registrar_csv_con_url(raiz):
    archivo = raiz / "datos.csv"
    archivo.write_text("a\n1\n")
    resultado = manifest.registrar(archivo, "https://example.invalid/datos.csv")
    assert resultado == {
        "ruta": "datos.csv",
        "sha256": manifest.sha256(archivo),
        "bytes": archivo.stat().st_size,
        "url_origen": "https://example.invalid/datos.csv",
    }


def test_registrar_parquet_cuenta_registros_y_omite_url(raiz):
    archivo = raiz / "datos.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(archivo)
    resultado = manifest.registrar(archivo)
    assert resultado["registros"] == 3
    assert "url_origen" not in resultado


def test_registrar_archivo_inexistente_falla(raiz):
    with pytest.raises(FileNotFoundError):
        manifest.registrar(raiz / "no_existe.csv")


def test_main_registra_todos_los_archivos(raiz, monkeypatch):
    url_nomenclatura = "https://example.invalid/nomenclatura.csv"
    monkeypatch.setattr(manifest, "MESES_URLS", {"2025_07": "https://example.invalid/m.csv"})
    monkeypatch.setattr(manifest, "NOMENCLATURA_URL", url_nomenclatura)
    (raiz / "data/raw").mkdir(parents=True)
    (raiz / "data/processed/viajes").mkdir(parents=True)
    (raiz / "data/raw/viajes_2025_07.csv").write_text("a\n1\n")
    (raiz / "data/raw" / Path(url_nomenclatura).name).write_text("a\n1\n")
    tabla = pd.DataFrame({"a": [1]})
    tabla.to_parquet(raiz / "data/processed/viajes/viajes_2025_07.parquet")
    tabla.to_parquet(raiz / "data/processed/estaciones.parquet")

    manifest.main()

    generado = json.loads((raiz / "data_manifest.json").read_text(encoding="utf-8"))
    assert len(generado["archivos"]) == 4
    assert {a["ruta"] for a in generado["archivos"] if "registros" in a} == {
        "data/processed/viajes/viajes_2025_07.parquet",
        "data/processed/estaciones.parquet",
    }


def test_main_falla_si_falta_un_archivo(raiz, monkeypatch):
    monkeypatch.setattr(manifest, "MESES_URLS", {"2025_07": "https://example.invalid/m.csv"})
    with pytest.raises(FileNotFoundError):
        manifest.main()
