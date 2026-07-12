import os
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Heart Disease Prediction")


def log_experiment(
    model,
    params,
    accuracy,
    precision,
    recall,
    roc_auc
):
    with mlflow.start_run():

        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        try:
            if not os.getenv("GITHUB_ACTIONS"):
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"MLflow model logging skipped: {e}")
