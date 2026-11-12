import numpy as np
import pandas as pd
import pytest

from kkbox_churn.config import (BINARY_COLS, COUNT_COLS, DAYS_COLS, FLOAT_COLS, ID_COL,
                                NOMINAL_NUM_COLS, NOMINAL_STR_COLS, OTHER_INT_COLS, TARGET)


def make_sample_data(n=1000, seed=0):
    """Clientes sintéticos con las mismas columnas y tipos que el dataset real."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({ID_COL: [f"client_{i}" for i in range(n)]})

    for col in NOMINAL_NUM_COLS:
        df[col] = rng.integers(1, 10, n).astype(float)
    for col in NOMINAL_STR_COLS:
        df[col] = rng.choice(["male", "female", None], n)
    for col in BINARY_COLS:
        df[col] = rng.integers(0, 2, n).astype(float)
    for col in COUNT_COLS:
        df[col] = rng.integers(0, 50, n).astype(float)
    for col in DAYS_COLS:
        df[col] = rng.integers(-100, 800, n).astype(float)
    for col in OTHER_INT_COLS:
        df[col] = rng.integers(0, 100, n).astype(float)   # incluye edades inválidas (< 13)
    for col in FLOAT_COLS:
        df[col] = rng.uniform(0, 1, n) if "rate" in col else rng.uniform(0, 200, n)

    # Valores ausentes, como en los datos reales
    for col in [*COUNT_COLS[:3], *FLOAT_COLS[:3], "age", "city"]:
        df.loc[rng.random(n) < 0.1, col] = np.nan

    # Target relacionado con una variable, para que el modelo aprenda algo
    churn_prob = np.where(df["auto_renew_rate"] < 0.3, 0.7, 0.1)
    df[TARGET] = (rng.random(n) < churn_prob).astype(int)

    return df


@pytest.fixture(scope="session")
def sample_df():
    return make_sample_data()


@pytest.fixture(scope="session")
def trained_model(sample_df):
    """Modelo real entrenado con los datos sintéticos (una vez por sesión)."""
    from kkbox_churn.models.train import train_model
    return train_model(sample_df)