from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from kkbox_churn.config import OHE_COLS, RS_COLS, PASSTHROUGH_COLS, RS_QUANTILE_RANGE
from kkbox_churn.preprocessing.preprocess import DataQualityTransformer


def build_preprocessor():
    """ColumnTransformer: one-hot, escalado robusto y binarias sin transformar."""
    return ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), OHE_COLS),
            ("rs", RobustScaler(quantile_range=RS_QUANTILE_RANGE), RS_COLS),
            ("bin", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",   # descarta msno, fechas y target
    )


def build_preprocessing_pipeline():
    """Limpieza + transformación de variables."""
    # Function Transformer: Medians are learned in fit() (train only) and reused in transform()
    return make_pipeline(DataQualityTransformer(), build_preprocessor())