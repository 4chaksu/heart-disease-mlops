from ucimlrepo import fetch_ucirepo


def load_data():

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    y = (y > 0).astype(int)
    y = y.values.ravel()
    return X, y
