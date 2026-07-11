# tests/test_evaluate.py

import numpy as np

from src.evaluate import evaluate_model


class MockModel:

    def predict(self, X):
        return np.array([0, 1, 1, 0])

    def predict_proba(self, X):
        return np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.7, 0.3]
            ]
        )


def test_evaluate_model():

    model = MockModel()

    X_test = np.zeros((4, 2))
    y_test = np.array([0, 1, 1, 0])

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
