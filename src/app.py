from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from src.model import get_model, MODEL_VERSION

app = FastAPI(
    title="Virtual Diabetese Triage API",
    version=MODEL_VERSION,
)

class DataStructure(BaseModel):
    age: float = Field(description = "Age in years", ge = 0, le = 120)
    sex: float = Field(description = "Sex indicator (1 = female, 2 = male)", ge = 1, le = 2) # kinda guessing tbh since the dataset doesn't specify, no info on it from the original paper the data comes from as well
    bmi: float = Field(description = "Body Mass Index", gt = 0, le = 70)
    bp: float = Field(description = "Mean arterial blood pressure",gt = 0)
    s1: float = Field(description = "tc (total serum cholesterol)", gt = 0)
    s2: float = Field(description = "ldl (low-density lipoproteins)", gt = 0)
    s3: float = Field(description = "hdl (high-density lipoproteins)", gt = 0)
    s4: float = Field(description = "tch (total cholesterol / HDL)", gt = 0)
    s5: float = Field(description = "ltg (possibly log of serum triglycerides level)", gt = 0)
    s6: float = Field(description = "glu (blood sugar level)", gt = 0) 

@app.post("/predict")
def predict(features: DataStructure):
    try:
        model = get_model()
        prediction = model.predict(features.dict())
        return {"prediction": prediction, "model_version": MODEL_VERSION}
    except ValueError as e:
        raise HTTPException(status_code = 422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code = 400, detail=str(e))

@app.get("/health", status_code = status.HTTP_200_OK)
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}