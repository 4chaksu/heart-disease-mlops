# tests/test_train_pipeline.py

import pandas as pd
from unittest.mock import Mock, patch

from src.train_pipeline import main


@patch("src.train_pipeline.joblib.dump")
@patch("src.train_pipeline.log_experiment")
@patch("src.train_pipeline.evaluate_model")
@patch("src.train_pipeline.train_random_forest")
@patch("src.train_pipeline.scale_features")
@patch("src.train_pipeline.split_data")
@patch("src.train_pipeline.encode_features")
@patch("src.train_pipeline.create_features")
@patch("src.train_pipeline.handle_outliers")
@patch("src.train_pipeline.handle_missing_values")
@patch("src.train_pipeline.load_data")
def test_main_pipeline(
    mock_load_data,
    mock_missing,
    mock_outliers,
    mock_features,
    mock_encode,
    mock_split,
    mock_scale,
    mock_train,
    mock_evaluate,
    mock_log,
    mock_dump,
):

    X = pd.DataFrame(
        {
            "age": [30, 40],
            "trestbps": [120, 130],
            "chol": [200, 220],
            "thalach": [150, 160],
            "oldpeak": [1.0, 2.0]
        }
    )

    y = [0, 1]

    mock_load_data.return_value = (
        X,
        y
    )

    mock_missing.return_value = X
    mock_outliers.return_value = X
    mock_features.return_value = X
    mock_encode.return_value = X

    mock_split.return_value = (
        X,
        X,
        y,
        y
    )

    scaler = Mock()

    mock_scale.return_value = (
        X,
        X,
        scaler
    )

    model = Mock()
    model.n_estimators = 100
    model.max_depth = 10
    model.min_samples_split = 2
    model.min_samples_leaf = 1

    mock_train.return_value = model

    mock_evaluate.return_value = {
        "accuracy": 0.90,
        "precision": 0.85,
        "recall": 0.88,
        "roc_auc": 0.91
    }

    main()

    mock_load_data.assert_called_once()
    mock_train.assert_called_once()
    mock_evaluate.assert_called_once()
    mock_log.assert_called_once()

    # model, scaler, feature_columns,
    # imputer_values
    assert mock_dump.call_count == 4