"""Monitoreo offline de data drift y calidad de datos para MiBici P1."""

from pathlib import Path
import re

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "evidently"


def load_samples():
    reference = pd.read_parquet(OUTPUT / "referencia_p1.parquet")
    current = pd.read_parquet(OUTPUT / "actual_p1.parquet")

    if list(reference.columns) != list(current.columns):
        raise ValueError("Las muestras tienen columnas diferentes.")

    return reference, current


def run_report(reference, current, exclude_month=False, filename=None):
    if exclude_month:
        reference = reference.drop(columns=["mes"])
        current = current.drop(columns=["mes"])

    definition = DataDefinition(numerical_columns=list(reference.columns))

    metrics = [DataDriftPreset()]
    if not exclude_month:
        metrics.append(DataSummaryPreset())

    result = Report(metrics).run(
        current_data=Dataset.from_pandas(
            current, data_definition=definition
        ),
        reference_data=Dataset.from_pandas(
            reference, data_definition=definition
        ),
    )

    if filename is None:
        filename = (
            "reporte_drift_p1_sin_mes.html"
            if exclude_month
            else "reporte_drift_p1.html"
        )
    result.save_html(str(OUTPUT / filename))

    return result


def export_drift_metrics(result, filename="resultados_drift_p1.csv"):
    rows = []

    for metric in result.dict()["metrics"]:
        name = metric["metric_name"]

        if not name.startswith("ValueDrift("):
            continue

        column = re.search(r"column=([^,]+)", name).group(1)
        method = re.search(r"method=(.*),threshold=", name).group(1)
        threshold = float(
            re.search(r"threshold=([0-9.]+)", name).group(1)
        )
        score = float(metric["value"])

        rows.append(
            {
                "variable": column,
                "metodo": method,
                "drift_score": score,
                "umbral": threshold,
                "drift_detectado": score > threshold,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / filename, index=False)
    return df


def export_quality_metrics(reference, current):
    rows = []

    for column in reference.columns:
        rows.append(
            {
                "variable": column,
                "faltantes_referencia": int(reference[column].isna().sum()),
                "faltantes_actual": int(current[column].isna().sum()),
                "porcentaje_referencia": round(
                    reference[column].isna().mean() * 100, 4
                ),
                "porcentaje_actual": round(
                    current[column].isna().mean() * 100, 4
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "resultados_calidad_p1.csv", index=False)
    return df


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    reference, current = load_samples()

    print("Muestras cargadas:")
    print(f"Referencia: {reference.shape}")
    print(f"Actual: {current.shape}")

    result = run_report(reference, current)
    drift = export_drift_metrics(result)
    quality = export_quality_metrics(reference, current)

    result_without_month = run_report(
        reference, current, exclude_month=True
    )

    summary = result.dict()["metrics"][0]["value"]
    summary_without_month = (
        result_without_month.dict()["metrics"][0]["value"]
    )

    print("\nResultados de Data Drift:")
    print(drift.to_string(index=False))
    print(
        f"\nDrift original: {summary['count']} de "
        f"{len(reference.columns)} variables"
    )
    print(
        f"Drift sin mes: {summary_without_month['count']} de "
        f"{len(reference.columns) - 1} variables"
    )

    print("\nCalidad de datos:")
    print(quality.to_string(index=False))

    seasonal_reference = pd.read_parquet(
        OUTPUT / "referencia_estacional_p1.parquet"
    )

    if list(seasonal_reference.columns) != list(current.columns):
        raise ValueError("La muestra estacional tiene columnas diferentes.")

    seasonal_result = run_report(
        seasonal_reference,
        current,
        filename="reporte_drift_p1_estacional.html",
    )

    seasonal_drift = export_drift_metrics(
        seasonal_result,
        filename="resultados_drift_p1_estacional.csv",
    )

    seasonal_summary = seasonal_result.dict()["metrics"][0]["value"]

    print("\nResultados de Data Drift estacional (jul-sep 2025 vs 2026):")
    print(seasonal_drift.to_string(index=False))
    print(
        f"\nDrift estacional: {seasonal_summary['count']} de "
        f"{len(seasonal_reference.columns)} variables"
    )

    print(f"\nReportes y CSV generados en: {OUTPUT}")


if __name__ == "__main__":
    main()
