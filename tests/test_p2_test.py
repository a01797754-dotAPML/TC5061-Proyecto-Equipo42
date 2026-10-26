"""Pruebas de la evaluación final de P2."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mibici_mlops.modeling.p2_test import evaluar_test_p2


@patch("mibici_mlops.modeling.p2_test.MlflowClient")
@patch("mibici_mlops.modeling.p2_test.mlflow.sklearn.load_model")
@patch("mibici_mlops.modeling.p2_test.cargar_o_preparar_p2")
@patch("mibici_mlops.modeling.p2_test.evaluar_predicciones")
def test_evaluacion_final_registra_metricas(
    mock_evaluar,
    mock_cache,
    mock_cargar_modelo,
    mock_client,
):
    """Verifica la evaluación y el registro de métricas de test."""
    cliente = mock_client.return_value
    cliente.get_run.return_value.data.tags = {
        "model_uri": "models:/modelo-prueba"
    }

    modelo = MagicMock()
    modelo.predict.return_value = [0, 1]
    mock_cargar_modelo.return_value = modelo

    X_test = pd.DataFrame({"variable": [1, 2]})
    y_test = pd.Series([0, 1])
    mock_cache.return_value = (
        None, None, None, None, X_test, y_test
    )

    mock_evaluar.return_value = {
        "test_macro_f1": 0.52,
        "test_accuracy": 0.96,
    }

    resultado = evaluar_test_p2("run-prueba")

    assert resultado["test_macro_f1"] == 0.52
    mock_cargar_modelo.assert_called_once_with(
        "models:/modelo-prueba"
    )
    mock_evaluar.assert_called_once()
    cliente.log_metric.assert_any_call(
        "run-prueba", "test_macro_f1", 0.52
    )


@patch("mibici_mlops.modeling.p2_test.MlflowClient")
def test_evaluacion_sin_modelo_falla(mock_client):
    """Impide evaluar un run sin modelo registrado."""
    mock_client.return_value.get_run.return_value.data.tags = {}

    with pytest.raises(ValueError, match="model_uri"):
        evaluar_test_p2("run-sin-modelo")
