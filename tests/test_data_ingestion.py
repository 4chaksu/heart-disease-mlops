# tests/test_data_ingestion.py
import pandas as pd
from unittest.mock import Mock, patch

from src.data_ingestion import load_data


@patch("src.data_ingestion.fetch_ucirepo")
def test_load_data(mock_fetch):

    mock_dataset = Mock()

    mock_dataset.data.features = pd.DataFrame(
        {
            "age": [50, 60],
            "chol": [200, 250]
        }
    )

    mock_dataset.data.targets = pd.DataFrame(
        {
            "target": [0, 2]
        }
    )

    mock_fetch.return_value = mock_dataset

    X, y = load_data()

    assert X.shape == (2, 2)
    assert list(y) == [0, 1]
