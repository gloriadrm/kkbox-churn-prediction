from pathlib import Path


# ============================================================
#  PATHS
# ============================================================
# config.py está en client_churn/src/kkbox_churn/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "02_data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "02_data" / "processed"

MODELS_DIR = PROJECT_ROOT / "04_models"
MODEL_PATH = MODELS_DIR / "execution_pipeline.joblib"

TRAIN_PATH = PROJECT_ROOT / "02_data" / "train" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "02_data" / "validation" / "validation.csv"

RAW_FILES = {
    "clients": "train_v2.csv",
    "members": "members_v3.csv",
    "transactions": "transactions.csv",
    "transactions_v2": "transactions_v2.csv",
}

# Parámetros de construcción del dataset
CUTOFF = "2017-03-01"
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ============================================================
#  TYPES 
# ============================================================

# IDENTIFICADOR Y TARGET
ID_COL = "msno"
TARGET = "is_churn"


# 1. VARIABLES DE TIPO FECHA
DATETIME_COLS = [
    "registration_init_time",
    "first_transaction",
    "last_transaction",
    "max_expire_date",
    "last_day_max_expire"
]


# 2. VARIABLES CATEGÓRICAS NOMINALES
NOMINAL_NUM_COLS = ["city", "registered_via"]   # códigos numéricos
NOMINAL_STR_COLS = ["gender"]                   # texto


# 3. VARIABLES ENTERAS

# Variables binarias
BINARY_COLS = [
    "last_day_any_cancel",
    "has_multi_non_cancelled_day",
    "has_multi_transaction_day",
    "last_day_all_auto_renew",
    "last_day_any_auto_renew",
    "has_mixed_cancel_day"
]

# Recuentos
COUNT_COLS = [
    "n_transaction_records",
    "n_transaction_days",
    "n_cancellations",
    "n_non_cancelled_transactions",
    "n_multi_transaction_days",
    "n_zero_list_price_transactions",
    "n_payment_methods",
    "n_distinct_plan_days"
]

# Variables temporales expresadas en días enteros
DAYS_COLS = [
    "membership_age_days",
    "max_plan_days",
    "recency_days",
    "transaction_span_days",
    "days_to_max_expiry",
    "days_to_expiry"
]

# Otras variables que representan cantidades enteras
OTHER_INT_COLS = [
    "age",
    "extra_transaction_records",
    "last_day_max_payment",
    "last_day_n_transactions",
    "max_transactions_same_day"
]

INT_COLS = BINARY_COLS + COUNT_COLS + DAYS_COLS + OTHER_INT_COLS


# 4. VARIABLES DECIMALES
FLOAT_COLS = [
    "auto_renew_rate",
    "avg_amount_paid",
    "avg_payment_per_30d",
    "avg_plan_days",
    "avg_transactions_per_day",
    "cancel_rate",
    "multi_transaction_day_rate",
    "payment_amount_std",
    "total_amount_paid",
    "total_non_cancelled_amount"
]

# ============================================================
# VALUE IMPUTATION
# ============================================================

ZERO_COLS = [
    "n_transaction_records",
    "n_transaction_days",
    "n_cancellations",
    "n_non_cancelled_transactions",
    "n_multi_transaction_days",
    "extra_transaction_records",
    "n_zero_list_price_transactions",
    "n_payment_methods",
    "n_distinct_plan_days",
    "total_amount_paid",
    "total_non_cancelled_amount",
    "auto_renew_rate",
    "cancel_rate",
    "multi_transaction_day_rate",
    "payment_amount_std"
]

# Variables cuya ausencia no equivale a cero
MEDIAN_COLS = [
    "age",
    "avg_amount_paid",
    "avg_payment_per_30d",
    "avg_plan_days",
    "avg_transactions_per_day",
    "days_to_expiry",
    "days_to_max_expiry",
    "recency_days",
    "transaction_span_days",
    "max_plan_days",
    "max_transactions_same_day",
    "last_day_n_transactions",
    "last_day_max_payment", 
    "membership_age_days"
]

# Rango de edad plausible
AGE_MIN, AGE_MAX = 18, 100


# ============================================================
# VARIABLE TRANSFORMATION
# ============================================================

# One-Hot Encoding: nominales + binarias con valor 2 = desconocido
OHE_COLS = NOMINAL_NUM_COLS + NOMINAL_STR_COLS + BINARY_COLS

# Binarias 0/1 que se conservan sin transformar
PASSTHROUGH_COLS = ["age_unknown"]

# Escalado robusto: todas las numéricas no binarias
RS_COLS = COUNT_COLS + DAYS_COLS + OTHER_INT_COLS + FLOAT_COLS

# Percentiles usados como rango en el RobustScaler
RS_QUANTILE_RANGE = (10.0, 90.0)

# ============================================================
# FEATURE SELECTION
# ============================================================

SELECTED_FEATURES = ['city', 'age', 'gender', 'registered_via',
       'registration_init_time', 'membership_age_days',
       'n_transaction_records', 'n_transaction_days', 'first_transaction',
       'last_transaction', 'n_cancellations', 'cancel_rate',
       'n_non_cancelled_transactions', 'avg_amount_paid', 'total_amount_paid',
       'payment_amount_std', 'total_non_cancelled_amount',
       'avg_payment_per_30d', 'avg_plan_days', 'max_plan_days',
       'n_distinct_plan_days', 'n_payment_methods',
       'n_zero_list_price_transactions', 'auto_renew_rate', 'max_expire_date',
       'max_transactions_same_day', 'avg_transactions_per_day',
       'n_multi_transaction_days', 'has_multi_non_cancelled_day',
       'has_mixed_cancel_day', 'last_day_n_transactions',
       'last_day_any_cancel', 'last_day_any_auto_renew',
       'last_day_all_auto_renew', 'last_day_max_expire',
       'last_day_max_payment', 'recency_days', 'transaction_span_days',
       'days_to_max_expiry', 'days_to_expiry', 'has_multi_transaction_day',
       'multi_transaction_day_rate', 'extra_transaction_records']

# ============================================================
# MODELO
# ============================================================

MODEL_PARAMS = {
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "max_iter": 1000,
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 200,
    "l2_regularization": 0,
    "random_state": 42,
}