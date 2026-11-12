import joblib
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

from kkbox_churn.config import MODEL_PARAMS, MODEL_PATH, SELECTED_FEATURES, TARGET
from kkbox_churn.preprocessing.pipeline import build_preprocessing_pipeline


def build_model():
    """Clasificador con los hiperparámetros elegidos en el grid search."""
    return HistGradientBoostingClassifier(**MODEL_PARAMS)


def build_training_pipeline():
    """Pipeline completo sin entrenar: limpieza + transformación + modelo."""
    return make_pipeline(build_preprocessing_pipeline(), build_model())


def split_features_target(df):
    """Separa predictores y target."""
    return df[SELECTED_FEATURES].copy(), df[TARGET].copy()


def train_model(df):
    """
    Entrena el pipeline y ajusta el umbral de clasificación (criterio de Youden)
    con validación cruzada sobre train. El umbral queda guardado en el modelo.
    """
    X, y = split_features_target(df)

    model = TunedThresholdClassifierCV(
        build_training_pipeline(),
        scoring="balanced_accuracy",   # equivale a maximizar Youden (TPR − FPR)
        cv=3,
    )
    model.fit(X, y)
    return model


def save_model(pipeline, path=MODEL_PATH):
    """Guarda el pipeline entrenado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path