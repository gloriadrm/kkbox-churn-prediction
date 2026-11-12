import logging
import joblib
import pandas as pd

from kkbox_churn.config import ID_COL, MODEL_PATH, SELECTED_FEATURES

logger = logging.getLogger(__name__)


def load_model(path=MODEL_PATH):
    """Carga el modelo entrenado (pipeline + umbral)."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo {path}. Ejecuta antes 'kkbox-churn train'.")
    model = joblib.load(path)
    logger.info("Modelo cargado (umbral %.4f)", model.best_threshold_)
    return model


def predict(model, df):
    """Devuelve, por cliente, la probabilidad de churn y la clase según el umbral del modelo."""
    missing = set(SELECTED_FEATURES + [ID_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en los datos de entrada: {sorted(missing)}")

    X = df[SELECTED_FEATURES]
    return pd.DataFrame({
        ID_COL: df[ID_COL].values,
        "churn_proba": model.predict_proba(X)[:, 1],
        "churn_pred": model.predict(X),
    })