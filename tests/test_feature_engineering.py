# tests/test_feature_engineering.py

import pandas as pd

from src.feature_engineering import create_features


def test_create_features():

    X = pd.DataFrame(
        {
            "age": [50],
            "thalach": [150],
            "exang": [1],
            "oldpeak": [2.0],
            "trestbps": [145],
            "chol": [250]
        }
    )

    result = create_features(X)

    assert "hr_utilization" in result.columns
    assert "exercise_risk" in result.columns
    assert "age_oldpeak" in result.columns
    assert "high_bp" in result.columns
    assert "high_chol" in result.columns

    assert result.loc[0, "exercise_risk"] == 2.0
    assert result.loc[0, "age_oldpeak"] == 100.0
    assert result.loc[0, "high_bp"] == 1
    assert result.loc[0, "high_chol"] == 1