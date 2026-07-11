import pandas as pd
from unittest.mock import Mock, patch

# Import after mocking if needed
from src.predict import predict


@patch("src.predict.handle_missing_values")
@patch("src.predict.create_features")
@patch("src.predict.encode_features")
def test_predict(
    mock_encode,
    mock_features,
    mock_missing,
):

    mock_missing.side_effect = lambda x: x
    mock_features.side_effect = lambda x: x
    mock_encode.side_effect = lambda x, y: x

    predict.feature_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    predict.scaler = Mock()
    predict.scaler.transform.side_effect = lambda x: x

    predict.model = Mock()
    predict.model.predict.return_value = [1]
    predict.model.predict_proba.return_value = [
        [0.2, 0.8]
    ]

    data = {
        "age": 50,
        "trestbps": 120,
        "chol": 240,
        "thalach": 150,
        "oldpeak": 1.0,
    }

    result = predict.predict(data)

    assert result["prediction"] == 1
    assert result["probability"] == 0.8