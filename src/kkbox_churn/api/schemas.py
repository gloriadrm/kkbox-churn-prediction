from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kkbox_churn.config import ID_COL, SELECTED_FEATURES

# Tipos reutilizables. Todos admiten None: el pipeline imputa los valores ausentes
Count = Annotated[int | None, Field(ge=0)]
Amount = Annotated[float | None, Field(ge=0)]
Rate = Annotated[float | None, Field(ge=0, le=1)]
Binary = Literal[0, 1] | None


class ClientFeatures(BaseModel):
    """Features de un cliente, tal como las genera build_features."""

    model_config = ConfigDict(
        extra="forbid",   # rechaza campos desconocidos (p. ej. un nombre mal escrito)
        json_schema_extra={"example": {
            "msno": "nFGjbH/PnsMzEArkxTIFc7vw1OV1kVsXSDDt6D0KWSU=",
            "city": 1, "age": 0, "gender": None, "registered_via": 7,
            "membership_age_days": 39,
            "n_transaction_records": 2, "n_transaction_days": 2,
            "n_cancellations": 0, "cancel_rate": 0.0, "n_non_cancelled_transactions": 2,
            "avg_amount_paid": 99.0, "total_amount_paid": 198.0, "payment_amount_std": 0.0,
            "total_non_cancelled_amount": 198.0, "avg_payment_per_30d": 99.0,
            "avg_plan_days": 30.0, "max_plan_days": 30, "n_distinct_plan_days": 1,
            "n_payment_methods": 1, "n_zero_list_price_transactions": 0,
            "auto_renew_rate": 1.0, 
            "max_transactions_same_day": 1, "avg_transactions_per_day": 1.0,
            "n_multi_transaction_days": 0, "has_multi_non_cancelled_day": 0,
            "has_mixed_cancel_day": 0, "last_day_n_transactions": 1,
            "last_day_any_cancel": 0, "last_day_any_auto_renew": 1,
            "last_day_max_payment": 99, "recency_days": 9, "transaction_span_days": 30,
            "days_to_max_expiry": 19, "days_to_expiry": 19,
            "has_multi_transaction_day": 0, "multi_transaction_day_rate": 0.0,
            "extra_transaction_records": 0,
        }},
    )

    # Identificador
    msno: str = Field(min_length=1, description="Identificador del cliente")

    # Perfil del miembro
    city: Annotated[int | None, Field(ge=1)] = None
    age: int | None = Field(None, description="Edad declarada; valores fuera de 13–100 se tratan como desconocidos")
    gender: Literal["male", "female"] | None = None
    registered_via: int | None = None
    membership_age_days: Count = None

    # Recuentos
    n_transaction_records: Count = None
    n_transaction_days: Count = None
    n_cancellations: Count = None
    n_non_cancelled_transactions: Count = None
    n_distinct_plan_days: Count = None
    n_payment_methods: Count = None
    n_zero_list_price_transactions: Count = None
    n_multi_transaction_days: Count = None
    max_transactions_same_day: Count = None
    last_day_n_transactions: Count = None
    extra_transaction_records: Count = None

    # Importes
    avg_amount_paid: Amount = None
    total_amount_paid: Amount = None
    payment_amount_std: Amount = None
    total_non_cancelled_amount: Amount = None
    avg_payment_per_30d: Amount = None
    last_day_max_payment: Amount = None

    # Planes y actividad
    avg_plan_days: Amount = None
    max_plan_days: Count = None
    avg_transactions_per_day: Amount = None

    # Tasas
    cancel_rate: Rate = None
    auto_renew_rate: Rate = None
    multi_transaction_day_rate: Rate = None

    # Indicadores binarios
    has_multi_non_cancelled_day: Binary = None
    has_mixed_cancel_day: Binary = None
    has_multi_transaction_day: Binary = None
    last_day_any_cancel: Binary = None
    last_day_any_auto_renew: Binary = None
    last_day_all_auto_renew: Binary = None

    # Tiempo (los días hasta la expiración pueden ser negativos: suscripción ya vencida)
    recency_days: Count = None
    transaction_span_days: Count = None
    days_to_max_expiry: int | None = None
    days_to_expiry: int | None = None


class PredictionResponse(BaseModel):
    msno: str
    churn_proba: float = Field(description="Probabilidad de churn (0–1)")
    churn_pred: Literal[0, 1] = Field(description="1 si la probabilidad supera el umbral del modelo")
    model_version: str


class BatchRequest(BaseModel):
    clients: list[ClientFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_version: str | None
    threshold: float | None


# El esquema debe recoger exactamente las features del modelo.
# Si se añade o quita una feature en config.py y no aquí, la API falla al arrancar.
_schema_features = set(ClientFeatures.model_fields) - {ID_COL}
if _schema_features != set(SELECTED_FEATURES):
    raise RuntimeError(
        "ClientFeatures no coincide con SELECTED_FEATURES. "
        f"Sobran: {_schema_features - set(SELECTED_FEATURES)} | "
        f"Faltan: {set(SELECTED_FEATURES) - _schema_features}"
    )


# Decisiones de diseño:

# extra="forbid": si alguien envía cancel_rte en vez de cancel_rate, la API responde con un error.
# Los tipos reutilizables (Count, Rate...) evitan repetir la misma validación en 40 campos.
# age no lleva rango: una edad de 0 o de 1000 no es un error de la petición. El pipeline ya la marca como desconocido y gestiona.
# La comprobación del final protege contra el error más probable en el futuro: cambiar las features del modelo y olvidarse de actualizar la API.
# Lotes de hasta 1.000 clientes, para que una petición enorme no bloquee la API. Para puntuar toda la cartera tenemos kkbox-churn predict.