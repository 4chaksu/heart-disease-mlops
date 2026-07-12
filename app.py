from fastapi import FastAPI
import logging
from prometheus_fastapi_instrumentator import Instrumentator

from src.predict import predict

logging.basicConfig(
    filename="api.log",
    level=logging.INFO
)

app = FastAPI()

# Prometheus Metrics
Instrumentator().instrument(app).expose(app)


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
