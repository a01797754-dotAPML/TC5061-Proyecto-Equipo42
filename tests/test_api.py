from fastapi.testclient import TestClient

from mibici_mlops.api.main import app, get_model


class FakeModel:
    def predict(self, data):
        return [2.5]


def override_get_model():
    return FakeModel()


app.dependency_overrides[get_model] = override_get_model

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "hora_dia": 8,
        "dia_semana": 2,
        "es_fin_semana": 0,
        "mes": 8,
        "capacidad": 20,
        "salidas_lag_1": 2,
        "salidas_lag_2": 3,
        "salidas_lag_24": 2,
        "salidas_lag_168": 4,
        "salidas_media_24h": 2.5,
        "salidas_media_4sem": 3.1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 2.5}


def test_predict_rechaza_hora_invalida():
    payload = {
        "hora_dia": 25,
        "dia_semana": 2,
        "es_fin_semana": 0,
        "mes": 8,
        "capacidad": 20,
        "salidas_lag_1": 2,
        "salidas_lag_2": 3,
        "salidas_lag_24": 2,
        "salidas_lag_168": 4,
        "salidas_media_24h": 2.5,
        "salidas_media_4sem": 3.1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_rechaza_campos_extra():
    payload = {
        "hora_dia": 8,
        "dia_semana": 2,
        "es_fin_semana": 0,
        "mes": 8,
        "capacidad": 20,
        "salidas_lag_1": 2,
        "salidas_lag_2": 3,
        "salidas_lag_24": 2,
        "salidas_lag_168": 4,
        "salidas_media_24h": 2.5,
        "salidas_media_4sem": 3.1,
        "campo_inventado": 999,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422