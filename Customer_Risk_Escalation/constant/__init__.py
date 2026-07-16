import os
from datetime import date

DATABASE_NAME = "customer_escalation"
TABLE_NAME = "escalation_dataset"
SQL_DB_URL_KEY = "SQL_DB_URL"
ARTIFACTS_DIR:str = "artifacts"
FILE_NAME:str = "supporttkts.csv"
CURRENT_YEAR = date.today().year

SCHEMA_FILE_PATH = os.path.join('Customer_Risk_Escalation','config','schema.yaml')


"""
Data Ingestion related constant.
"""
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_DATA_STORE_DIR:str = "data_store"


"""
Data Validation related constant.
"""
DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_REPORT_DIR:str = "reports"
DATA_VALIDATION_REPORT_FILE_NAME:str = "validation_report.yaml"
DATA_VALIDATION_STATUS_FILE:str = "validation_status.txt"
DATA_VALIDATION_DRIFT_DASHBOARD_NAME:str = "drift_dashboard.html"


"""
Data Transformation related constant.
"""
DATA_TRANSFORMATION_DIR_NAME:str = "data_transformation"
DATA_TRANSFORMATION_TRAIN_DIR:str = os.path.join("transformed", "train")
DATA_TRANSFORMATION_TEST_DIR:str = os.path.join("transformed", "test")
DATA_TRANSFORMATION_SCALER_DIR:str = os.path.join("transformed", "scaler")

X_TRAIN_FILE_NAME:str = "X_train.csv"
X_TEST_FILE_NAME:str = "X_test.csv"
Y_TRAIN_FILE_NAME:str = "y_train.csv"
Y_TEST_FILE_NAME:str = "y_test.csv"
TEXT_TRAIN_FILE_NAME:str = "text_train.csv"
TEXT_TEST_FILE_NAME:str = "text_test.csv"
SCALER_FILE_NAME:str = "scaler.pkl"

TEST_SIZE:float = 0.2
RANDOM_STATE:int = 42
SENTINEL_VALUE:int = -1
UNKNOWN_FILL:str = "Unknown"
UNRESOLVED_FILL:str = "unresolved"
RESOLUTION_TIME_THRESHOLD:int = 120


"""
Tabular Data Model Training related constant.
"""
MODEL_TRAINER_DIR_NAME:str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str = "trained_model"
MODEL_TRAINER_MODEL_FILE_NAME:str = "model.pkl"

MODEL_TRAINER_EXPECTED_RECALL:float = 0.80
MODEL_TRAINER_EXPECTED_AUC:float = 0.75

MLFLOW_TRACKING_URI:str = "file:///D:/End-to-end-ML/Customer-Risk-Escalation-Engine/mlruns"
MLFLOW_EXPERIMENT_NAME:str = "customer_escalation_tabular1"


"""
NLP Data Model Training related constant.
"""
NLP_DIR_NAME:str = "nlp_trainer"
NLP_EMBEDDINGS_DIR:str = "embeddings"

NLP_TRAIN_EMBEDDINGS_FILE:str = "train_embeddings.npy"
NLP_TEST_EMBEDDINGS_FILE:str = "test_embeddings.npy"

DISTILBERT_MODEL_NAME:str = "distilbert-base-uncased"
MLFLOW_EXPERIMENT_NLP:str = "customer_escalation_nlp1"
NLP_MAX_LENGTH:int = 128
NLP_BATCH_SIZE:int = 32

"""
Fusion Model Training related constant.
"""
FUSION_DIR_NAME:str = "late_fusion"
FUSION_MODEL_DIR:str = "fusion_model"
FUSION_MODEL_FILE_NAME:str = "fusion_model.pkl"
MLFLOW_EXPERIMENT_FUSION:str = "customer_escalation_fusion1"
FUSION_EXPECTED_RECALL:float = 0.80