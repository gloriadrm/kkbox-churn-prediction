import numpy as np
import pandas as pd
import pytest

from kkbox_churn.config import BINARY_COLS, MEDIAN_COLS, SELECTED_FEATURES, ZERO_COLS
from kkbox_churn.preprocessing.preprocess import DataQualityTransformer, cast_types, prepare_age


def test_prepare_age_marks_invalid_ages():
    df = pd.DataFrame({"age": pd.array([5, 30, None, 150], dtype="Int32")})
    out = prepare_age(df)

    assert out["age_unknown"].tolist() == [1, 0, 1, 1]
    assert out["age"].isna().tolist() == [True, False, True, True]


def test_cast_types_rejects_missing_column(sample_df):
    X = sample_df[SELECTED_FEATURES].drop(columns="cancel_rate")
    with pytest.raises(ValueError, match="no existen"):
        cast_types(X)


def test_cast_types_rejects_unknown_column(sample_df):
    X = sample_df[SELECTED_FEATURES].assign(unexpected=1)
    with pytest.raises(ValueError, match="sin tipar"):
        cast_types(X)


def test_transformer_leaves_no_missing_values(sample_df):
    out = DataQualityTransformer().fit_transform(sample_df[SELECTED_FEATURES])
    assert out[ZERO_COLS + MEDIAN_COLS + BINARY_COLS].isna().sum().sum() == 0


def test_transformer_imputes_with_train_medians(sample_df):
    """Los datos nuevos se imputan con las medianas aprendidas en fit (sin data leakage)."""
    X = sample_df[SELECTED_FEATURES]
    transformer = DataQualityTransformer().fit(X)

    new_client = X.head(1).copy()
    new_client[MEDIAN_COLS] = np.nan
    out = transformer.transform(new_client)

    for col in MEDIAN_COLS:
        assert out[col].iloc[0] == transformer.medians_[col]