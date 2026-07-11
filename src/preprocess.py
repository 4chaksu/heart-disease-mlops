import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(X):

    X = X.copy()

    for col in X.columns:

        if X[col].isnull().any():

            if pd.api.types.is_numeric_dtype(X[col]):

                fill_value = X[col].median()

                if pd.notna(fill_value):
                    X[col] = X[col].fillna(fill_value)

            else:

                if not X[col].mode().empty:
                    X[col] = X[col].fillna(
                        X[col].mode().iloc[0]
                    )

    return X


def handle_outliers(X, numerical_features):

    X = X.copy()

    for col in numerical_features:

        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        X[col] = X[col].clip(lower, upper)

    return X


def encode_features(X, categorical_features):

    X = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=True
    )

    bool_cols = X.select_dtypes(include="bool").columns

    X[bool_cols] = X[bool_cols].astype(int)

    return X


def scale_features(X_train, X_test, numerical_features):

    scaler = StandardScaler()

    X_train[numerical_features] = scaler.fit_transform(
        X_train[numerical_features]
    )

    X_test[numerical_features] = scaler.transform(
        X_test[numerical_features]
    )

    return X_train, X_test, scaler
