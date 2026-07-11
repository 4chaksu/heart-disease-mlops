from src.data_ingestion import load_data

from src.preprocess import (
    handle_missing_values,
    handle_outliers,
    encode_features,
    scale_features
)

from src.feature_engineering import create_features

from src.train import (
    split_data,
    train_random_forest
)

from src.evaluate import evaluate_model

from src.mlflow_tracking import log_experiment

import joblib
from pathlib import Path
import joblib

def main():

    print("Loading data...")
    X, y = load_data()


    print("Handling missing values...")
    X = handle_missing_values(X)

    numerical_features = [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak'
    ]

    categorical_features = [
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'ca',
        'thal'
    ]

    print("Handling outliers...")
    X = handle_outliers(
        X,
        numerical_features
    )

    print("Creating engineered features...")
    X = create_features(X)

    print("Encoding categorical features...")
    X = encode_features(
        X,
        categorical_features
    )

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("Scaling numerical features...")
    X_train, X_test, scaler = scale_features(
        X_train,
        X_test,
        numerical_features
    )

    print("Training Random Forest...")
    model = train_random_forest(
        X_train,
        y_train
    )

    print("Evaluating model...")
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("Logging experiment to MLflow...")

    params = {
        "model": "RandomForest",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf
    }

    log_experiment(
        model=model,
        params=params,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        roc_auc=metrics["roc_auc"]
    )
    
    # Learn values from training data
    imputer_values = {}

    for col in X.columns:

        if X[col].isnull().any():

            if X[col].dtype in ["int64", "float64"]:
                imputer_values[col] = X[col].median()
            else:
                imputer_values[col] = X[col].mode().iloc[0]


    print("Saving model artifacts...")

    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "heart_disease_model.pkl"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.pkl"
    )

    joblib.dump(
        X_train.columns.tolist(),
        MODELS_DIR / "feature_columns.pkl"
    )

    joblib.dump(
        imputer_values,
        MODELS_DIR / "imputer_values.pkl"
    )
    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
