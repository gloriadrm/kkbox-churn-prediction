import logging
from datetime import datetime

import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline

from kkbox_churn.config import (CUTOFF, MODEL_PARAMS, SELECTED_FEATURES, TARGET,
                                MLFLOW_MODEL_NAME, SELECTION_METRIC)
from kkbox_churn.models.evaluate import evaluate
from kkbox_churn.models.tracking import setup_mlflow, promote_if_better, SKOPS_TRUSTED_TYPES
from kkbox_churn.models.tracking import setup_mlflow, promote_if_better
from kkbox_churn.preprocessing.pipeline import build_preprocessing_pipeline

logger = logging.getLogger(__name__)

THRESHOLD_CV = 3



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
        cv=THRESHOLD_CV,
    )
    model.fit(X, y)
    return model


def train_and_log(train_df, val_df):
    """
    Entrena, evalúa en validación y registra todo en MLflow.
    Devuelve el modelo, las métricas, la versión registrada y si se promovió a champion.
    """
    setup_mlflow()
    run_name = f"hgb_{datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name):
        # Entrenamiento
        model = train_model(train_df)

        # Evaluación en validación
        X_val, y_val = split_features_target(val_df)
        proba = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)
        metrics = evaluate(y_val, proba, pred)

        # Parámetros
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_params({
            "model_type": "HistGradientBoostingClassifier",
            "threshold_criterion": "youden (balanced_accuracy)",
            "threshold_cv": THRESHOLD_CV,
            "cutoff": CUTOFF,
            "n_features": len(SELECTED_FEATURES),
            "n_train": len(train_df),
            "n_val": len(val_df),
        })

        # Métricas
        mlflow.log_metric("threshold", model.best_threshold_)
        mlflow.log_metrics({f"val_{name}": value for name, value in metrics.items()})

        # Modelo + registro de una nueva versión
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_val.head(100), pred[:100]),
            input_example=X_val.head(5),
            registered_model_name=MLFLOW_MODEL_NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        version = info.registered_model_version
        promoted = promote_if_better(version, metrics[SELECTION_METRIC])
        mlflow.set_tag("promoted_to_champion", promoted)

    return model, metrics, version, promoted