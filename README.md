# Virtual Diabetes Clinic Triage API

This project is a small, containerized ML service to predict short-term diabetes progression. It's built as part of a university MLOps assignment.

The service provides a "risk score" (a continuous value) that a virtual clinic can use to prioritize patient follow-ups.

## Iteration Plan

* **v0.1 (Magikarp 🎏):** Baseline model using `StandardScaler` + `LinearRegression`.
* **v0.2 (Gyarados 🐉):** *[In Progress]* Improved model (e.g., `RandomForestRegressor`) to achieve a better RMSE.

---

## API Endpoints
### TODO
## How to Run

You can run the service locally (for development) or using the pre-built Docker container (for production/grading).

### Option 1: Run with Docker (Recommended)

This pulls the pre-built image from the GitHub Container Registry (GHCR) and runs it.

1.  **Pull the image (e.g., v0.1):**
    ```bash
    docker pull ghcr.io/heishigan/virtual-diabetes-triage:v0.1
    ```

2.  **Run the container:**
    ```bash
    docker run -d --rm -p 8000:8000 --name diabetes-api ghcr.io/heishigan/virtual-diabetes-triage:v0.1
    ```
    The API is now available at `http://localhost:8000`. Go to `http://localhost:8000/docs` to access UI version.

### Option 2: Run Locally (For Development)

This method requires you to train the model first to generate the local artifact files.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Heishigan/virtual-diabetes-triage.git
    cd virtual-diabetes-triage
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3.  **Train the model (e.g., v0.1):**
    This creates the model files in the `./release_assets` directory.
    ```bash
    python src/train.py --model-version "v0.1" --output-path "./release_assets"
    ```

4.  **Set environment variables and run the app:**
    This tells the app where to find the local model files.

    **On Windows (PowerShell):**
    ```powershell
    $env:MODEL_VERSION = "v0.1"
    $env:MODEL_PATH = "release_assets\model-v0.1.joblib"
    $env:SCALER_PATH = "release_assets\scaler-v0.1.joblib"
    
    uvicorn src.app:app --host localhost --port 8000
    ```

    **On macOS/Linux (Bash):**
    ```bash
    export MODEL_VERSION="v0.1"
    export MODEL_PATH="release_assets/model-v0.1.joblib"
    export SCALER_PATH="release_assets/scaler-v0.1.joblib"
    
    uvicorn src.app:app --host localhost --port 8000
    ```
    The API is now available at `http://localhost:8000`. Go to `http://localhost:8000/docs` to access UI version.

---

## Test the API
### TODO