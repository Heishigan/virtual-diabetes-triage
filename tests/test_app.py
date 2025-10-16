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
def test_predict_success():
    valid_payload = {
        "age": 0.038076,
        "sex": 0.050680,
        "bmi": 0.061696,
        "bp": 0.021872,
        "s1": -0.044223,
        "s2": -0.034821,
        "s3": -0.043401,
        "s4": -0.002592,
        "s5": 0.019908,
        "s6": -0.017646,
    }

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    payload = response.json()
    
    # Check prediction is a float in reasonable range
    assert isinstance(payload["prediction"], float)
    assert 50 < payload["prediction"] < 350
    assert "model_version" in payload

# Test /predict returns 422 when required field is missing
def test_predict_missing_field_returns_422():
    missing_payload = {
        "age": 0.038076,
        "sex": 0.050680,
        "bmi": 0.061696,
        "bp": 0.021872,
        "s1": -0.044223,
        "s2": -0.034821,
        "s3": -0.043401,
        "s4": -0.002592,
        "s5": 0.019908,
        # "s6" missing
    }

    response = client.post("/predict", json=missing_payload)
    assert response.status_code == 422

# Test /predict returns 422 when validation fails due to invalid field types
def test_predict_invalid_types_return_422():
    invalid_payload = {
        "age": "not_a_number",  # Should be float
        "sex": 0.050680,
        "bmi": 0.061696,
        "bp": 0.021872,
        "s1": -0.044223,
        "s2": -0.034821,
        "s3": -0.043401,
        "s4": -0.002592,
        "s5": 0.019908,
        "s6": -0.017646,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422