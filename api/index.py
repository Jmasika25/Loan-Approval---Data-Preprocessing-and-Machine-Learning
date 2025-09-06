from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained pipeline (must be in repo root or /api folder)
model = joblib.load("model_pipeline.pkl")

app = FastAPI()

# Input format for prediction
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "HR ML API is running on Vercel!"}

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return {"prediction": int(prediction)}
