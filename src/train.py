import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import json

# Set seeds for reproducibility
np.random.seed(42)

def explore_data():
    """Explore the dataset to understand its structure"""
    data = load_diabetes(as_frame=True)
    print("=== Dataset Overview ===")
    print(f"Dataset shape: {data.frame.shape}")
    print(f"Feature names: {data.feature_names}")
    print(f"Target name: target")
    
    # Display first few rows
    print("\n=== First 5 rows ===")
    print(data.frame.head())
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(data.frame.describe())
    
    return data

def load_data():
    """Load and prepare the diabetes dataset"""
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]
    return X, y

def train_baseline_model():
    """Train v0.1 baseline model"""
    print("Loading and exploring data...")
    data = explore_data()
    
    # Load data for training
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Baseline Model RMSE: {rmse:.4f}")
    
    # Save artifacts
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    
    # Save metrics
    metrics = {
        "rmse": float(rmse), 
        "model_version": "v0.1",
        "model_type": "LinearRegression",
        "dataset_info": {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "feature_names": list(X.columns)
        }
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Model training completed! Artifacts saved.")
    return rmse

if __name__ == "__main__":
    train_baseline_model()