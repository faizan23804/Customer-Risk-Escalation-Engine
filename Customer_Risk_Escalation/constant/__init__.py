import os
import sys
import numpy as np # type: ignore
from datetime import date

DATABASE_NAME = "customer_escalation"
TABLE_NAME = "escalation_dataset"
SQL_DB_URL_KEY = "SQL_DB_URL"
ARTIFACTS_DIR:str = "artifacts"
FILE_NAME:str = "supporttkts.csv"
CURRENT_YEAR = date.today().year

"""
Data Ingestion related constant.
"""
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_DATA_STORE_DIR:str = "data_store"


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