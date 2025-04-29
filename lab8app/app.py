from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

app = FastAPI()

# Load model from local path
model_path = "/Users/dhvanipatel/Desktop/MLOps/mlruns/3/3dcf4e790ab44ae592dc887b92de711a/artifacts/model"
model = mlflow.sklearn.load_model(model_path)

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
async def predict(data: Features):
    input_array = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}