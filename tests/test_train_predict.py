import pytest

from kkbox_churn.config import ID_COL
from kkbox_churn.models.predict import predict


def test_model_has_valid_threshold(trained_model):
    assert 0 < trained_model.best_threshold_ < 1


def test_predict_output(trained_model, sample_df):
    preds = predict(trained_model, sample_df)

    assert list(preds.columns) == [ID_COL, "churn_proba", "churn_pred"]
    assert len(preds) == len(sample_df)
    assert preds["churn_proba"].between(0, 1).all()
    assert set(preds["churn_pred"]) <= {0, 1}


def test_predict_applies_model_threshold(trained_model, sample_df):
    preds = predict(trained_model, sample_df)
    expected = (preds["churn_proba"] >= trained_model.best_threshold_).astype(int)
    assert (preds["churn_pred"] == expected).all()


def test_predict_rejects_missing_features(trained_model, sample_df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        predict(trained_model, sample_df.drop(columns="recency_days"))