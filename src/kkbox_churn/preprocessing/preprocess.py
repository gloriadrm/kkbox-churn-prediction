import numpy as np
import pandas as pd
import cloudpickle
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from kkbox_churn.config import (PROJECT_ROOT, 
                        ID_COL, TARGET, DATETIME_COLS, NOMINAL_NUM_COLS,
                        NOMINAL_STR_COLS, INT_COLS, FLOAT_COLS,
                        BINARY_COLS, ZERO_COLS, MEDIAN_COLS, 
                        AGE_MIN, AGE_MAX, 
                        SELECTED_FEATURES, 
                        OHE_COLS, PASSTHROUGH_COLS, RS_COLS, RS_QUANTILE_RANGE )

### Instantiate data cleaning function
def cast_types(df):
    """
    Asigna los tipos de datos a cada grupo de variables.
    Devuelve una copia del DataFrame.
    """
    df = df.copy()

    # Comprobaciones previas
    all_cols = (DATETIME_COLS + NOMINAL_NUM_COLS + NOMINAL_STR_COLS
                + INT_COLS + FLOAT_COLS)

    duplicated = {c for c in all_cols if all_cols.count(c) > 1}
    assert not duplicated, f"Columnas en más de un grupo: {duplicated}"

    missing = set(all_cols) - set(df.columns)
    assert not missing, f"Columnas que no existen en el df: {missing}"

    remaining = set(df.columns) - set(all_cols) - {ID_COL, TARGET}
    assert not remaining, f"Columnas sin tipar: {remaining}"

    # Conversión de tipos
    df[DATETIME_COLS] = df[DATETIME_COLS].apply(pd.to_datetime)
    df[NOMINAL_NUM_COLS] = df[NOMINAL_NUM_COLS].astype("Int32").astype("string")
    df[NOMINAL_STR_COLS] = df[NOMINAL_STR_COLS].astype("string")
    df[INT_COLS] = df[INT_COLS].astype("Int32")
    df[FLOAT_COLS] = df[FLOAT_COLS].astype("float64")

    # El target solo existe en train y validación
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype("Int8")

    return df

def prepare_age(df):
    """Marca edades ausentes o fuera de rango y las convierte en NaN."""
    df = df.copy()
    valid_age = df["age"].between(AGE_MIN, AGE_MAX).fillna(False).astype(bool)
    # Variable que recoge la ausencia de info
    df["age_unknown"] = (~valid_age).astype("int8")
    df["age"] = df["age"].where(valid_age)
    return df


def fit_medians(df):
    """Calcula las medianas en train. Se ejecuta solo una vez."""
    df = prepare_age(cast_types(df))
    return {
        col: round(df[col].median()) if pd.api.types.is_integer_dtype(df[col])
        else df[col].median()
        for col in MEDIAN_COLS
    }

def clean_data(df, medians):
    """Limpia e imputa usando las medianas aprendidas en train."""
    temp = cast_types(df)
    temp = prepare_age(temp)

    # Nominales
    nominal_cols = NOMINAL_NUM_COLS + NOMINAL_STR_COLS
    temp[nominal_cols] = temp[nominal_cols].fillna("Unknown")

    # Binarias (2 = desconocido)
    temp[BINARY_COLS] = temp[BINARY_COLS].fillna(2)

    # Numéricas
    temp[ZERO_COLS] = temp[ZERO_COLS].fillna(0)
    temp[MEDIAN_COLS] = temp[MEDIAN_COLS].fillna(medians)

    # Comprobación final
    imputed_cols = nominal_cols + BINARY_COLS + ZERO_COLS + MEDIAN_COLS
    remaining_na = temp[imputed_cols].isna().sum()
    assert remaining_na.sum() == 0, f"Quedan NaN: {remaining_na[remaining_na > 0].to_dict()}"

    return temp

# Instantiate the data quality transformer
class DataQualityTransformer(BaseEstimator, TransformerMixin):
    """Tipado, limpieza e imputación con medianas aprendidas en fit."""

    def fit(self, X, y=None):
        self.medians_ = fit_medians(X)
        return self

    def transform(self, X):
        return clean_data(X, self.medians_)

