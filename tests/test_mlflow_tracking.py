from unittest.mock import Mock, patch

from src.mlflow_tracking import log_experiment


@patch("src.mlflow_tracking.mlflow.sklearn.log_model")
@patch("src.mlflow_tracking.mlflow.log_metric")
@patch("src.mlflow_tracking.mlflow.log_params")
@patch("src.mlflow_tracking.mlflow.start_run")
def test_log_experiment(
    mock_start_run,
    mock_log_params,
    mock_log_metric,
    mock_log_model,
):
    mock_start_run.return_value.__enter__ = Mock()
    mock_start_run.return_value.__exit__ = Mock()

    model = Mock()
    params = {"n_estimators": 100}

    log_experiment(
        model,
        params,
        accuracy=0.90,
        precision=0.85,
        recall=0.80,
        roc_auc=0.92,
    )

    mock_log_params.assert_called_once_with(params)

    assert mock_log_metric.call_count == 4

    mock_log_metric.assert_any_call("accuracy", 0.90)
    mock_log_metric.assert_any_call("precision", 0.85)
    mock_log_metric.assert_any_call("recall", 0.80)
    mock_log_metric.assert_any_call("roc_auc", 0.92)

    mock_log_model.assert_called_once_with(model, "model")