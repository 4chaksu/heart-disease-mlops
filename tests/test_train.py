# tests/test_train.py

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.train import (
    split_data,
    train_logistic,
    train_random_forest
)


def test_split_data():

    X = pd.DataFrame(
        {
            "age": range(20)
        }
    )

    y = np.array(
        [0] * 10 + [1] * 10
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_train_logistic():

    X_train = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "chol": [180, 200, 220, 240]
        }
    )

    y_train = [0, 0, 1, 1]

    model = train_logistic(
        X_train,
        y_train
    )

    assert model is not None

    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


@patch("src.train.RandomizedSearchCV")
def test_train_random_forest(
    mock_random_search
):

    mock_best_model = Mock()

    mock_search_instance = Mock()
    mock_search_instance.best_estimator_ = (
        mock_best_model
    )

    mock_random_search.return_value = (
        mock_search_instance
    )

    X_train = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "chol": [180, 200, 220, 240]
        }
    )

    y_train = [0, 0, 1, 1]

    model = train_random_forest(
        X_train,
        y_train
    )

    mock_search_instance.fit.assert_called_once()

    assert model == mock_best_model
