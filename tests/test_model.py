import joblib

def test_model_loaded():

    model = joblib.load("models/heart_disease_model.pkl")

    assert model is not None