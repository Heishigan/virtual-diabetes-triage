import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import types
import pytest
import model as model_module

class FakeSklearnModel:
    def predict(self, X):
        return [42.0]

def _mock_exists_factory(always_exists: bool = True):
    def _mock_exists(path):
        return always_exists
    return _mock_exists

def _mock_joblib_load(path):
    path_str = str(path)
    if path_str.endswith("model.joblib"):
        return FakeSklearnModel()
    raise RuntimeError(f"Unexpected load path in test: {path}")

def _valid_features():
    return {
        "age": 0.038,
        "sex": 0.051,
        "bmi": 0.062,
        "bp": 0.022,
        "s1": -0.044,
        "s2": -0.035,
        "s3": -0.043,
        "s4": -0.003,
        "s5": 0.020,
        "s6": -0.018,
    }

def test_datamodel_predict_returns_float(monkeypatch):
    monkeypatch.setattr(model_module.os.path, "exists", _mock_exists_factory(True))
    monkeypatch.setattr(model_module.joblib, "load", _mock_joblib_load)

    dm = model_module.DataModel()
    result = dm.predict(_valid_features())
    assert isinstance(result, float)
    assert result == 42.0

def test_datamodel_predict_raises_on_missing_features(monkeypatch):
    monkeypatch.setattr(model_module.os.path, "exists", _mock_exists_factory(True))
    monkeypatch.setattr(model_module.joblib, "load", _mock_joblib_load)

    dm = model_module.DataModel()
    features = _valid_features()
    features.pop("s6")
    with pytest.raises(ValueError) as err:
        dm.predict(features)
    assert "Missing required features" in str(err.value)

def test_get_model_singleton(monkeypatch):
    monkeypatch.setattr(model_module, "_model_instance", None, raising=True)
    monkeypatch.setattr(model_module.os.path, "exists", _mock_exists_factory(True))
    monkeypatch.setattr(model_module.joblib, "load", _mock_joblib_load)

    m1 = model_module.get_model()
    m2 = model_module.get_model()
    assert m1 is m2