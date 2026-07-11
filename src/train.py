from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_logistic(X_train, y_train):

    model = LogisticRegression()

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    param_dist = {
        "n_estimators":[100,200,300,500],
        "max_depth":[None,5,10,15,20],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4],
        "max_features":["sqrt","log2"]
    }

    rf = RandomForestClassifier(
        random_state=42
    )

    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    random_search.fit(
        X_train,
        y_train
    )

    return random_search.best_estimator_
