import joblib
import pandas as pd
from pathlib import Path

from src.preprocess import (
    handle_missing_values,
    encode_features
)

from src.feature_engineering import create_features

# Load artifacts
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "heart_disease_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
feature_columns = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")

numerical_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


def predict(data):

    # Convert input dictionary to dataframe
    df = pd.DataFrame([data])

    # Same preprocessing as training
    df = handle_missing_values(df)

    # Same feature engineering
    df = create_features(df)

    # Same encoding
    df = encode_features(
        df,
        categorical_features
    )

    # add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # remove extra columns
    df = df[feature_columns]
    
    # Use saved scaler
    df[numerical_features] = scaler.transform(
        df[numerical_features]
    )

    # Predict
    prediction = model.predict(df)

    # Probability
    probability = model.predict_proba(df)

    return {
        "prediction": int(prediction[0]),
        "probability": float(
            probability[0][prediction[0]]
        )
    } 

