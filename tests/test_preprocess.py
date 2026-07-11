# tests/test_preprocess.py

import pandas as pd
import numpy as np

from src.preprocess import (
    handle_missing_values,
    handle_outliers,
    encode_features,
    scale_features,
)


def test_handle_missing_values():

    df = pd.DataFrame(
        {
            "age": [20, np.nan, 40],
            "sex": ["M", np.nan, "F"],
        }
    )

    result = handle_missing_values(df)

    # Median of [20, 40] = 30
    assert result.loc[1, "age"] == 30

    # Mode = "F" or "M" depending on pandas ordering
    assert result["sex"].isnull().sum() == 0


def test_handle_missing_values_no_nulls():

    df = pd.DataFrame(
        {
            "age": [20, 30, 40]
        }
    )

    result = handle_missing_values(df)

    pd.testing.assert_frame_equal(df, result)


def test_handle_outliers():

    df = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 500]
        }
    )

    result = handle_outliers(df, ["age"])

    assert result["age"].max() < 500


def test_handle_outliers_no_changes():

    df = pd.DataFrame(
        {
            "age": [20, 25, 30, 35]
        }
    )

    result = handle_outliers(df, ["age"])

    assert len(result) == 4


def test_encode_features():

    df = pd.DataFrame(
        {
            "sex": [0, 1],
            "cp": [1, 2],
        }
    )

    result = encode_features(
        df,
        ["sex", "cp"]
    )

    assert result.shape[1] > 0


def test_encode_features_boolean_conversion():

    df = pd.DataFrame(
        {
            "flag": [True, False, True]
        }
    )

    result = encode_features(df, [])

    assert result["flag"].dtype == int


def test_scale_features():

    X_train = pd.DataFrame(
        {
            "age": [20, 30, 40]
        }
    )

    X_test = pd.DataFrame(
        {
            "age": [25, 35]
        }
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test,
        ["age"]
    )

    assert round(X_train_scaled["age"].mean(), 6) == 0
    assert scaler is not None


def test_scale_features_shape():

    X_train = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "chol": [200, 250, 300]
        }
    )

    X_test = pd.DataFrame(
        {
            "age": [25, 35],
            "chol": [220, 280]
        }
    )

    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train,
        X_test,
        ["age", "chol"]
    )

    assert X_train_scaled.shape == (3, 2)
    assert X_test_scaled.shape == (2, 2)