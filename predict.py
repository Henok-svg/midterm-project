from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="California Housing Predictor")
model = joblib.load('model.joblib')

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float

@app.get("/")
def home():
    return {
        "message": "California Housing Price Predictor",
        "docs": "Visit /docs for interactive Swagger UI"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != len(FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURES)} features, got {len(request.features)}"
        )
    prediction = model.predict(np.array(request.features).reshape(1, -1))
    return PredictionResponse(prediction=float(prediction[0]))