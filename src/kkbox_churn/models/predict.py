import joblib
import pandas as pd

from kkbox_churn.config import ID_COL, MODEL_PATH, SELECTED_FEATURES


def load_model(path=MODEL_PATH):
    """Carga el modelo entrenado (pipeline + umbral)."""
    return joblib.load(path)


def predict(model, df):
    """Devuelve, por cliente, la probabilidad de churn y la clase según el umbral del modelo."""
    X = df[SELECTED_FEATURES]

    return pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "churn_proba": model.predict_proba(X)[:, 1],
        "churn_pred": model.predict(X),   # ya aplica best_threshold_
    })