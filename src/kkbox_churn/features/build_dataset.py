import pandas as pd
import numpy as np
from kkbox_churn.features.members import build_member_features
from kkbox_churn.features.transactions import build_transaction_features

def integrate_transactions(tx1, tx2): 
    """
    Integra transactions y transactions_v2. 
    Aún no filtramos por ninguna fecha solo unimos dataframes transaccionales.
    """
    # Comprobamos que ambos datasets tengan mismas columnas 
    assert set(tx1.columns) == set(tx2.columns), "Las columnas de los dfs no coinciden."

    # Eliminamos dupliados 
    tx1 = tx1.drop_duplicates()
    tx2 = tx2.drop_duplicates()
    
    # Concatenamos los históricos filtrados
    txs_all = pd.concat(
        [tx1, tx2],
        ignore_index=True
    )

    return txs_all

def build_features(clients, members, transactions, cutoff):
    """
    Integra las features de miembros y transacciones.
    Devuelve una fila por cliente.
    """
    member_features = build_member_features(members, cutoff)

    transaction_features = build_transaction_features(
        transactions, clients, cutoff
    )

    df = (
        clients
        .merge(
            member_features,
            on="msno",
            how="left",
            validate="one_to_one"
        )
        .merge(
            transaction_features,
            on="msno",
            how="left",
            validate="one_to_one"
        )
    )

    assert len(df) == len(clients)
    assert df["msno"].is_unique

    return df