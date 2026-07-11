from fastapi import FastAPI
import logging

from src.predict import predict

logging.basicConfig(
    filename="api.log",
    level=logging.INFO
)

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API"
    }

@app.post("/predict")
def make_prediction(data: dict):

    logging.info("Prediction request received")

    result = predict(data)

    return result