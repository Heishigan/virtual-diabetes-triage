import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
import joblib
import json
import os
import argparse

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
    
    # Check if data is already scaled
    print("\n=== Feature Statistics (checking for standardization) ===")
    stats = data.frame.drop(columns=['target']).describe()
    print(stats.loc[['mean', 'std']])
    print("\nNOTE: Features are already standardized (mean ≈ 0, std ≈ 1)")
    print("    No additional StandardScaler needed!\n")
    
    return data

def load_data():
    """Load and prepare the diabetes dataset"""
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]
    return X, y

def calibrate_high_risk_threshold(y_true, y_pred, percentile=75):
    """
    Calibrate high-risk flag based on predicted progression score.
    
    Args:
        y_true: Actual progression values
        y_pred: Predicted progression values
        percentile: Percentile threshold for high-risk classification (default: 75th = top 25%)
    
    Returns:
        dict: Threshold value and classification metrics
    """
    # Determine threshold from predictions (e.g., 75th percentile = top 25% highest risk)
    threshold = np.percentile(y_pred, percentile)
    
    # Create binary classifications
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    # Calculate how many would be flagged
    n_flagged = y_pred_binary.sum()
    n_actual_high_risk = y_true_binary.sum()
    
    return {
        "threshold": float(threshold),
        "percentile": percentile,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_flagged": int(n_flagged),
        "n_actual_high_risk": int(n_actual_high_risk)
    }

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model with multiple metrics including high-risk calibration"""
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation RMSE (5-fold)
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='neg_root_mean_squared_error'
    )
    cv_rmse = -cv_scores.mean()
    
    # High-risk flag calibration (75th percentile = top 25%)
    high_risk_metrics = calibrate_high_risk_threshold(y_test, y_pred, percentile=75)
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"RMSE (test):           {rmse:.4f}")
    print(f"MAE (test):            {mae:.4f}")
    print(f"R² (test):             {r2:.4f}")
    print(f"CV RMSE (5-fold):      {cv_rmse:.4f} (±{cv_scores.std():.4f})")
    print(f"\nHigh-Risk Flag Calibration (threshold={high_risk_metrics['threshold']:.2f}):")
    print(f"  Precision:           {high_risk_metrics['precision']:.4f}")
    print(f"  Recall:              {high_risk_metrics['recall']:.4f}")
    print(f"  F1-Score:            {high_risk_metrics['f1_score']:.4f}")
    print(f"  Patients flagged:    {high_risk_metrics['n_flagged']}/{len(y_test)}")
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "cv_rmse": float(cv_rmse),
        "cv_rmse_std": float(cv_scores.std()),
        "high_risk_calibration": high_risk_metrics
    }

def train_model(model_version, output_path):
    """Train and compare multiple models for v0.2"""
    print("="*60)
    print("Virtual Diabetes Triage - Model Training v0.2")
    print("="*60)
    
    print("\nLoading and exploring data...")
    explore_data()
    
    # Load data for training
    X, y = load_data()
    
    # Split data (same split as v0.1 for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set:     {X_test.shape}")
    print(f"\nData is already standardized - using features directly!")
    
    # Compare multiple models
    models_to_test = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    for name, model in models_to_test.items():
        results[name] = evaluate_model(
            model, X_train, X_test, y_train, y_test, name
        )
    
    # Select best model based on RMSE
    best_model_name = min(results, key=lambda k: results[k]["rmse"])
    best_rmse = results[best_model_name]["rmse"]
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name} (RMSE: {best_rmse:.4f})")
    print(f"{'='*60}")
    
    # Train final model on best algorithm
    final_model = models_to_test[best_model_name]
    final_model.fit(X_train, y_train)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save artifacts with version naming
    model_filename = f"model-{model_version}.joblib"
    metrics_filename = f"metrics-{model_version}.json"
    
    # Save model (NO SCALER NEEDED!)
    joblib.dump(final_model, os.path.join(output_path, model_filename))
    print(f"\nModel saved: {model_filename}")
    
    # Save comprehensive metrics
    metrics = {
        "model_version": model_version,
        "best_model": best_model_name,
        "rmse": results[best_model_name]["rmse"],
        "mae": results[best_model_name]["mae"],
        "r2": results[best_model_name]["r2"],
        "cv_rmse": results[best_model_name]["cv_rmse"],
        "cv_rmse_std": results[best_model_name]["cv_rmse_std"],
        "high_risk_calibration": results[best_model_name]["high_risk_calibration"],
        "all_models_comparison": results,
        "dataset_info": {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
            "data_already_scaled": True
        },
        "improvements_vs_v01": {
            "removed_double_standardization": True,
            "models_tested": list(models_to_test.keys()),
            "added_high_risk_flag": True
        }
    }
    
    metrics_path = os.path.join(output_path, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved: {metrics_filename}")
    print("\nModel training completed successfully!")
    
    return best_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    train_model(model_version=args.model_version, output_path=args.output_path)