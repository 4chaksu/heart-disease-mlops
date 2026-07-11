def create_features(X):

    X = X.copy()

    X["hr_utilization"] = X["thalach"] / (
        220 - X["age"]
    )

    X["exercise_risk"] = (
        X["exang"] * X["oldpeak"]
    )

    X["age_oldpeak"] = (
        X["age"] * X["oldpeak"]
    )

    X["high_bp"] = (
        X["trestbps"] >= 140
    ).astype(int)

    X["high_chol"] = (
        X["chol"] >= 240
    ).astype(int)

    return X
