
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
 
def build_transaction_features(transactions, clients, cutoff):
    """
    Construye features históricas por cliente utilizando
    únicamente transacciones anteriores al cutoff.
    """

    # Fecha de corte según período a predecir (prevención data leakage)
    cutoff = pd.Timestamp(cutoff)

    # Convertir a formato datetime las columnas de tipo fecha
    date_cols = transactions.columns[transactions.columns.str.contains('date')]
    
    transactions[date_cols] = transactions[date_cols].apply(
        lambda c: pd.to_datetime(c.astype("Int64").astype(str), format="%Y%m%d", errors= "raise")
    )

    # Filtrar por clientes (archivo train) y por fecha inferior al período a predecir
    tx = transactions.loc[
        transactions["msno"].isin(clients["msno"])
        & (transactions["transaction_date"] < cutoff)
    ].copy()

    # ----------------------------
    # 2. Variables auxiliares
    # ----------------------------
    
    # Operaciones no canceladas
    tx["non_cancelled"] = (tx["is_cancel"] == 0).astype(int)
    
    # Importe asociado a operaciones no canceladas
    tx["non_cancelled_amount"] = np.where(
        tx["is_cancel"] == 0,
        tx["actual_amount_paid"],
        0
    )
    
    # Indicador de operaciones con precio de catálogo cero
    tx["zero_list_price"] = (tx["plan_list_price"] == 0).astype(int)
    
    # Precio normalizado por 30 días.
    # Evitamos divisiones por cero cuando payment_plan_days = 0.
    tx["payment_per_30d"] = np.where(
        tx["payment_plan_days"] > 0,
        tx["actual_amount_paid"] * 30 /
        tx["payment_plan_days"].replace(0, np.nan),
        np.nan
    )

    # ----------------------------
    # 3. Agregaciones por cliente
    # ----------------------------
    
    tx_features = tx.groupby("msno").agg(

    # Actividad transaccional
    n_transaction_records =("transaction_date", "size"),
    n_transaction_days=("transaction_date", "nunique"),
    first_transaction=("transaction_date", "min"),
    last_transaction=("transaction_date", "max"),

    # Cancelaciones
    n_cancellations=("is_cancel", "sum"),
    cancel_rate=("is_cancel", "mean"),
    n_non_cancelled_transactions=("non_cancelled", "sum"),
    
    # Importes registrados
    avg_amount_paid=("actual_amount_paid", "mean"),
    total_amount_paid=("actual_amount_paid", "sum"),
    payment_amount_std=("actual_amount_paid", "std"),
    total_non_cancelled_amount=("non_cancelled_amount", "sum"),

    # Precio normalizado
    avg_payment_per_30d=("payment_per_30d", "mean"),
    
    # Planes contratados
    avg_plan_days=("payment_plan_days", "mean"),
    max_plan_days=("payment_plan_days", "max"),
    n_distinct_plan_days=("payment_plan_days", "nunique"),
    n_payment_methods=("payment_method_id", "nunique"),
    n_zero_list_price_transactions=("zero_list_price", "sum"),

    # Renovación automática
    auto_renew_rate=("is_auto_renew", "mean"),

    # Vencimiento máximo registrado
    max_expire_date=("membership_expire_date", "max")
    )
    
    # ----------------------------
    # 4. Actividad por cliente y día
    # ----------------------------
        
    daily_activity = (
        tx.groupby(["msno", "transaction_date"])
        .agg(
            n_daily_transactions=("is_cancel", "size"),
            n_daily_cancellations=("is_cancel", "sum"),
            n_daily_non_cancelled=("non_cancelled", "sum")
        )
        .reset_index()
    )
    
    # Indicadores diarios
    daily_activity["multi_transaction_day"] = (
        daily_activity["n_daily_transactions"] > 1
    ).astype(int)
    
    daily_activity["multi_non_cancelled_day"] = (
        daily_activity["n_daily_non_cancelled"] >= 2
    ).astype(int)
    
    daily_activity["multi_mixed_cancel_day"] = (
        (daily_activity["n_daily_cancellations"] >= 1) &
        (daily_activity["n_daily_non_cancelled"] >= 1)
    ).astype(int)
    
    # Agregación final por cliente
    daily_features = daily_activity.groupby("msno").agg(
        max_transactions_same_day=("n_daily_transactions", "max"),
        avg_transactions_per_day=("n_daily_transactions", "mean"),
        n_multi_transaction_days=("multi_transaction_day", "sum"),
        has_multi_non_cancelled_day=("multi_non_cancelled_day", "max"),
        has_mixed_cancel_day=("multi_mixed_cancel_day", "max")
    )

    # --------------------------------------------------
    # 5. Variables de la última fecha observada
    # --------------------------------------------------
    
    # Identificamos todas las operaciones del último día de cada cliente
    last_dates = tx.groupby("msno")["transaction_date"].transform("max")
    
    last_day_tx = tx.loc[tx["transaction_date"].eq(last_dates)]
    
    last_day_features = last_day_tx.groupby("msno").agg(
        last_day_n_transactions=("transaction_date", "size"),
        last_day_any_cancel=("is_cancel", "max"),
        last_day_any_auto_renew=("is_auto_renew", "max"),
        last_day_all_auto_renew=("is_auto_renew", "min"),
        last_day_max_expire=("membership_expire_date", "max"),
        last_day_max_payment=("actual_amount_paid", "max")
    )

    # -------------------------------------------
    # 6. Integración de todas las features
    # -------------------------------------------
    
    client_txs = (
        tx_features
        .join(daily_features, how="left")
        .join(last_day_features, how="left")
    )

    # -------------------------------------------
    # 7. Variables temporales
    # -------------------------------------------
    
    # Días desde la última transacción
    client_txs["recency_days"] = (
        cutoff - client_txs["last_transaction"]
    ).dt.days
    
    # Periodo observado entre primera y última transacción
    client_txs["transaction_span_days"] = (
        client_txs["last_transaction"] -
        client_txs["first_transaction"]
    ).dt.days
    
    # Días hasta el vencimiento máximo registrado
    client_txs["days_to_max_expiry"] = (
        client_txs["max_expire_date"] - cutoff
    ).dt.days
    
    # Días hasta el vencimiento máximo del último día observado
    client_txs["days_to_expiry"] = (
        client_txs["last_day_max_expire"] - cutoff
    ).dt.days
    
    
    # Identifica si el cliente ha tenido alguna vez operaciones múltiples en una fecha
    client_txs["has_multi_transaction_day"] = (
        client_txs["max_transactions_same_day"] > 1
    ).astype(int)
    
    # Proporción de días de actividad con múltiples transacciones
    client_txs["multi_transaction_day_rate"] = (
        client_txs["n_multi_transaction_days"] /
        client_txs["n_transaction_days"]
    )
    
    # Cuantifica número de registros adicionales respecto a una operación por día
    client_txs["extra_transaction_records"] = (
        client_txs["n_transaction_records"] -
        client_txs["n_transaction_days"]
    )

    # -------------------------------------------
    # 8. Resultado final
    # -------------------------------------------
    client_txs = client_txs.reset_index()
    
    # Comprobación: una única fila por cliente
    if not client_txs["msno"].is_unique:
        raise ValueError("Hay clientes duplicados.")
    
    # Comprobación: no se incorporan transacciones de marzo
    if tx["transaction_date"].max() >= cutoff:
    raise ValueError(
        "Hay transacciones posteriores a la fecha de corte: riesgo de data leakage."
    )
    
    logger.info("Features de transacciones: %s clientes, %s columnas", *client_txs.shape)

    return client_txs