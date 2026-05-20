

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


SAMPLE_PAYLOAD = {
    "DER_mass_MMC": 138.4,
    "DER_mass_transverse_met_lep": 51.6,
    "DER_mass_vis": 97.8,
    "PRI_tau_pt": 32.6,
    "PRI_lep_pt": 44.1,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "model_name" in data
    assert "model_version" in data
    assert data["prediction"] in ["signal", "background"]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0

def test_batch_predict_endpoint():
    response = client.post("/batch-predict", json=[SAMPLE_PAYLOAD, SAMPLE_PAYLOAD])

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "model_name" in prediction
        assert "model_version" in prediction
        assert prediction["prediction"] in ["signal", "background"]
        assert isinstance(prediction["probability"], float)
        assert 0.0 <= prediction["probability"] <= 1.0

def test_predict_endpoint_validation_error():
    invalid_payload = {
        "DER_mass_MMC": 138.4
    }

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422