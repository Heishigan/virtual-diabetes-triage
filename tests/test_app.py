import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient
import app as app_module

client = TestClient(app_module.app)

class FakeModel:
    def predict(self, features: dict):
        return 111.11

# Test /health endpoint returns 'ok' and contains model version
def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_version" in payload

# Test /predict endpoint returns correct prediction with valid input
def test_predict_success(monkeypatch):
    monkeypatch.setattr(app_module, "get_model", lambda: FakeModel())

    valid_payload = {
        "age": 50,
        "sex": 1,
        "bmi": 25.0,
        "bp": 80.0,
        "s1": 150,
        "s2": 100.0,
        "s3": 55.0,
        "s4": 3.0,
        "s5": 4.5,
        "s6": 120,
    }

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 111.11
    assert "model_version" in payload

# Test /predict returns 422 when required field is missing
def test_predict_missing_field_returns_422(monkeypatch):
    monkeypatch.setattr(app_module, "get_model", lambda: FakeModel())

    missing_payload = {
        "age": 50,
        "sex": 1,
        "bmi": 25.0,
        "bp": 80.0,
        "s1": 150,
        "s2": 100.0,
        "s3": 55.0,
        "s4": 3.0,
        "s5": 4.5,
        # "s6" missing
    }

    response = client.post("/predict", json=missing_payload)
    assert response.status_code == 422

# Test /predict returns 422 when validation fails due to invalid values
def test_predict_invalid_values_return_422(monkeypatch):
    monkeypatch.setattr(app_module, "get_model", lambda: FakeModel())

    invalid_payload = {
        "age": -1,  # invalid: ge=0
        "sex": 3,   # invalid: le=2
        "bmi": -5.0,  # invalid: gt=0
        "bp": -10.0,  # invalid: gt=0
        "s1": -1,   # invalid: gt=0
        "s2": -1.0, # invalid: gt=0
        "s3": -1.0, # invalid: gt=0
        "s4": -1.0, # invalid: gt=0
        "s5": -1.0, # invalid: gt=0
        "s6": -1,   # invalid: gt=0
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
