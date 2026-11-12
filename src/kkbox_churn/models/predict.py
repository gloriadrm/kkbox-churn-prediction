import logging
import pandas as pd

from kkbox_churn.config import ID_COL, SELECTED_FEATURES
from kkbox_churn.models.tracking import load_champion

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def load_model():
    """Carga el modelo champion desde MLflow."""
    model, version = load_champion()
    logger.info("Modelo champion cargado (versión %s, umbral %.4f)", version, model.best_threshold_)
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