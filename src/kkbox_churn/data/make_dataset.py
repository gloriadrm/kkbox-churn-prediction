from sklearn.model_selection import train_test_split

from kkbox_churn.config import (CUTOFF, TEST_SIZE, RANDOM_STATE, TARGET,
                                TRAIN_PATH, VALIDATION_PATH)
from kkbox_churn.data.load import load_raw_data, save_dataset
from kkbox_churn.features.build_dataset import build_features, integrate_transactions


def make_dataset(cutoff=CUTOFF):
    """Construye el dataset de features por cliente a partir de los datos en bruto."""
    raw = load_raw_data()
    transactions = integrate_transactions(raw["transactions"], raw["transactions_v2"])

    return build_features(
        clients=raw["clients"],
        members=raw["members"],
        transactions=transactions,
        cutoff=cutoff,
    )


def prepare_data(cutoff=CUTOFF):
    """Construye el dataset, lo divide de forma estratificada y guarda train y validación."""
    df = make_dataset(cutoff)

    train, val = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )

    save_dataset(train, TRAIN_PATH)
    save_dataset(val, VALIDATION_PATH)
    return train, val