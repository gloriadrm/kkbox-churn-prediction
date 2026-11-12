import logging
import pandas as pd

from kkbox_churn.config import RAW_DATA_DIR, RAW_FILES

logger = logging.getLogger(__name__)


def load_raw_data():
    """Carga las tablas en bruto. Devuelve un diccionario {nombre: DataFrame}."""
    missing = [f for f in RAW_FILES.values() if not (RAW_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan archivos en {RAW_DATA_DIR}: {missing}. "
            "Descárgalos del reto KKBox Churn Prediction de Kaggle."
        )

    data = {}
    for name, file in RAW_FILES.items():
        data[name] = pd.read_csv(RAW_DATA_DIR / file)
        logger.info("Cargado %s: %s filas", file, f"{len(data[name]):,}")
    return data


def load_dataset(path):
    """Carga un dataset ya construido (train, validación o datos nuevos)."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo {path}.")
    return pd.read_csv(path)

def save_dataset(df, path):
    """Guarda un dataset creando la carpeta si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path