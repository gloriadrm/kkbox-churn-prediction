import pandas as pd

from kkbox_churn.config import RAW_DATA_DIR, RAW_FILES


def load_raw_data():
    """Carga las tablas en bruto. Devuelve un diccionario {nombre: DataFrame}."""
    return {name: pd.read_csv(RAW_DATA_DIR / file) for name, file in RAW_FILES.items()}


def load_dataset(path):
    """Carga un dataset ya construido (train, validación o datos nuevos)."""
    return pd.read_csv(path)


def save_dataset(df, path):
    """Guarda un dataset creando la carpeta si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path