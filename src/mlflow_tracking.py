import mlflow
import mlflow.sklearn


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

        mlflow.sklearn.log_model(model, "model")
