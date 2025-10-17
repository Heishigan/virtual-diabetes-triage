import os
import joblib
# import numpy as np
import pandas as pd
import json

# ARTIFACTS_DIR = "artifacts"
# MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
# METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")

# MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
# SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
# METRICS_PATH = os.path.join(METRICS_DIR, "metrics.json")

MODEL_PATH = "/app/model.joblib"
SCALER_PATH = "/app/scaler.joblib"

# try:
#     with open(METRICS_PATH, 'r') as f:
#         metrics = json.load(f)
#     MODEL_VERSION = metrics.get("model_version", "version_not_found")
# except FileNotFoundError:
#     MODEL_VERSION = "v0.0_uninitialized"
# except json.JSONDecodeError:
#     MODEL_VERSION = "v0.0_error"

FEATURE_NAMES = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

class DataModel:
    def __init__(self, model_path = MODEL_PATH, scaler_path = SCALER_PATH):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model/scaler not found at {model_path} or {scaler_path}. Check Dockerfile.")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, features: dict):
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X_input = pd.DataFrame([features[f] for f in FEATURE_NAMES], index = FEATURE_NAMES).T
        X_scaled = self.scaler.transform(X_input)
        y_pred = self.model.predict(X_scaled)
        # print(X_scaled) # scales seem off
        return float(y_pred[0])

_model_instance = None
def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = DataModel()
    return _model_instance
