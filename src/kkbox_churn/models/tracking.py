import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from kkbox_churn.config import (MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_DIR, MLFLOW_EXPERIMENT,
                                MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS, SELECTION_METRIC)

# Tipos del modelo que skops debe aceptar al guardar y cargar.
# Son seguros porque el modelo lo entrenamos nosotros.
SKOPS_TRUSTED_TYPES = [
    "kkbox_churn.preprocessing.preprocess.DataQualityTransformer",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.metrics._classification.balanced_accuracy_score",
    "sklearn.metrics._scorer._CurveScorer",
    "sklearn.utils._metadata_requests.MetadataRequest",
    "sklearn.utils._metadata_requests.MethodMetadataRequest",
]

logger = logging.getLogger(__name__)


def setup_mlflow():
    """Conecta con el servidor de tracking y crea el experimento si no existe."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        # En local se fija la carpeta de artefactos; en remoto la gestiona el servidor
        is_local = MLFLOW_TRACKING_URI.startswith("sqlite")
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=MLFLOW_ARTIFACT_DIR.as_uri() if is_local else None,
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def get_champion_score(client):
    """Métrica de validación del champion actual, o None si aún no hay ninguno."""
    try:
        version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS)
    except MlflowException:
        return None
    return client.get_run(version.run_id).data.metrics.get(f"val_{SELECTION_METRIC}")


def promote_if_better(version, score, force=False):
    """Asigna el alias champion a la nueva versión si mejora al champion actual (o si se fuerza)."""
    client = MlflowClient()
    current = get_champion_score(client)

    if force or current is None or score > current:
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS, version)
        logger.info("Versión %s promovida a champion (%s %.4f, anterior %s%s)",
                    version, SELECTION_METRIC, score,
                    "ninguno" if current is None else f"{current:.4f}",
                    ", promoción forzada" if force else "")
        return True

    logger.info("Versión %s no promovida: %s %.4f no mejora al champion (%.4f)",
                version, SELECTION_METRIC, score, current)
    return False


def load_champion():
    """Carga el modelo champion desde el Model Registry. Devuelve (modelo, versión)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS)
    except MlflowException as e:
        raise LookupError(
            f"No hay ningún modelo con alias '{MLFLOW_CHAMPION_ALIAS}'. "
            "Ejecuta antes 'kkbox-churn train'."
        ) from e

    # Se carga por número de versión: así modelo y versión siempre coinciden
    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/{version.version}")
    return model, version.version