import os
import joblib
import pandas as pd

MODEL_VERSION = os.getenv("MODEL_VERSION", "local-dev")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.joblib")

FEATURE_NAMES = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

class DataModel:
    """
    Model wrapper for diabetes progression prediction.
    
    v0.2 Changes:
    - Removed StandardScaler dependency (data already scaled)
    - Direct prediction on input features
    - Supports multiple model types (LinearRegression, Ridge, RandomForest)
    """
    
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Check Dockerfile and build process."
            )
        
        self.model = joblib.load(model_path)
        print(f"Model loaded successfully: {type(self.model).__name__}")

    def predict(self, features: dict):
        """
        Predict diabetes progression score from patient features.
        
        Args:
            features: Dictionary with standardized feature values
                     (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
        
        Returns:
            float: Predicted progression score (higher = greater risk)
        
        Raises:
            ValueError: If required features are missing
        """
        # Validate input
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Prepare input DataFrame (features already scaled in dataset)
        X_input = pd.DataFrame(
            [features[f] for f in FEATURE_NAMES], 
            index=FEATURE_NAMES
        ).T
        
        # Direct prediction (no scaler needed - data already standardized!)
        y_pred = self.model.predict(X_input)
        
        return float(y_pred[0])

# Singleton instance for efficient loading
_model_instance = None

def get_model():
    """Get or create singleton model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = DataModel()
    return _model_instance