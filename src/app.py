from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from model import get_model, MODEL_VERSION

app = FastAPI(
    title = "Virtual Diabetese Triage API",
    version = MODEL_VERSION,
)

class DataStructure(BaseModel):
    age: float = Field(description="Standardized age")
    sex: float = Field(description="Standardized sex indicator")
    bmi: float = Field(description="Standardized Body Mass Index")
    bp: float = Field(description="Standardized mean arterial blood pressure")
    s1: float = Field(description="Standardized tc (total serum cholesterol)")
    s2: float = Field(description="Standardized ldl (low-density lipoproteins)")
    s3: float = Field(description="Standardized hdl (high-density lipoproteins)")
    s4: float = Field(description="Standardized tch (total cholesterol / HDL)")
    s5: float = Field(description="Standardized ltg (log of serum triglycerides)")
    s6: float = Field(description="Standardized glu (blood sugar level)")

@app.post("/predict")
def predict(features: DataStructure):
    try:
        model = get_model()
        prediction = model.predict(features.model_dump())
        return {"prediction": prediction, "model_version": MODEL_VERSION}
    except ValueError as e:
        raise HTTPException(status_code = 422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code = 400, detail=str(e))

@app.get("/health", status_code = status.HTTP_200_OK)
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}