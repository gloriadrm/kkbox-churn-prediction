import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from kkbox_churn.api.schemas import (BatchRequest, BatchResponse, ClientFeatures,
                                     HealthResponse, PredictionResponse)
from kkbox_churn.logging_config import setup_logging
from kkbox_churn.models.predict import predict
from kkbox_churn.models.tracking import load_champion
from kkbox_churn.config import ID_COL, SELECTED_FEATURES, MAX_MISSING_FEATURES_RATIO

logger = logging.getLogger(__name__)

def _warn_missing_features(clients: list[ClientFeatures]) -> None:
    """Registra un aviso por cada cliente con demasiadas features ausentes."""
    for client in clients:
        missing = [f for f in SELECTED_FEATURES if getattr(client, f) is None]
        ratio = len(missing) / len(SELECTED_FEATURES)

        if ratio > MAX_MISSING_FEATURES_RATIO:
            logger.warning(
                "Cliente %s con %d de %d features ausentes (%.0f %%); "
                "la predicción se basa sobre todo en valores imputados",
                getattr(client, ID_COL), len(missing), len(SELECTED_FEATURES), 100 * ratio,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar la API."""
    setup_logging()
    try:
        app.state.model, app.state.model_version = load_champion()
        logger.info("Modelo champion cargado (versión %s)", app.state.model_version)
    except Exception:
        # La API arranca igualmente: /health informará de que no hay modelo
        logger.exception("No se pudo cargar el modelo champion")
        app.state.model, app.state.model_version = None, None
    yield


app = FastAPI(
    title="KKBox Churn API",
    description="Predicción de abandono de suscriptores de KKBox con el modelo champion registrado en MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)


def _run_prediction(request: Request, clients: list[ClientFeatures]) -> list[PredictionResponse]:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    _warn_missing_features(clients)

    df = pd.DataFrame([client.model_dump() for client in clients])

    try:
        preds = predict(model, df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    version = request.app.state.model_version
    return [PredictionResponse(**row, model_version=version) for row in preds.to_dict("records")]


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health(request: Request):
    """Estado de la API y del modelo cargado."""
    model = request.app.state.model
    return HealthResponse(
        status="ok" if model is not None else "unavailable",
        model_version=request.app.state.model_version,
        threshold=model.best_threshold_ if model is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict_one(client: ClientFeatures, request: Request):
    """Predice el riesgo de churn de un cliente."""
    return _run_prediction(request, [client])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predicción"])
def predict_batch(body: BatchRequest, request: Request):
    """Predice el riesgo de churn de hasta 1.000 clientes."""
    return BatchResponse(predictions=_run_prediction(request, body.clients))


# Decisiones de diseño:

# lifespan: el modelo se descarga de DagsHub una sola vez al arrancar, no en cada petición.
# La API arranca aunque no haya modelo: /health devolvería unavailable y /predict respondería 503. Así, en Docker se puede ver qué falla.
# Endpoints con def y no con async def: la predicción usa CPU (scikit-learn) y no espera a la red. Con def, FastAPI la ejecuta en un hilo aparte y la API sigue atendiendo otras peticiones mientras tanto.
# Códigos HTTP: 422 para datos de entrada incorrectos (FastAPI lo usa también para los errores de Pydantic) y 503 para “servicio no disponible”.