import pytest
from fastapi.testclient import TestClient

import kkbox_churn.api.main as api
from kkbox_churn.api.schemas import ClientFeatures

EXAMPLE = ClientFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture
def client(trained_model, monkeypatch):
    """API con el modelo de prueba en lugar del champion de DagsHub."""
    monkeypatch.setattr(api, "load_champion", lambda: (trained_model, "test"))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] == "test"


def test_predict_example(client):
    response = client.post("/predict", json=EXAMPLE)
    body = response.json()

    assert response.status_code == 200
    assert body["msno"] == EXAMPLE["msno"]
    assert 0 <= body["churn_proba"] <= 1
    assert body["churn_pred"] in (0, 1)


def test_predict_only_id(client):
    """Un cliente sin ningún dato se predice con valores imputados."""
    assert client.post("/predict", json={"msno": "test"}).status_code == 200


def test_predict_rejects_out_of_range(client):
    assert client.post("/predict", json={**EXAMPLE, "cancel_rate": 1.5}).status_code == 422


def test_predict_rejects_unknown_field(client):
    assert client.post("/predict", json={**EXAMPLE, "unknown": 1}).status_code == 422


def test_predict_batch(client):
    clients = [{**EXAMPLE, "msno": f"c{i}"} for i in range(3)]
    response = client.post("/predict/batch", json={"clients": clients})

    assert response.status_code == 200
    assert [p["msno"] for p in response.json()["predictions"]] == ["c0", "c1", "c2"]


def test_api_without_model_returns_503(monkeypatch):
    def fail():
        raise LookupError("sin modelo")

    monkeypatch.setattr(api, "load_champion", fail)
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "unavailable"
        assert c.post("/predict", json=EXAMPLE).status_code == 503