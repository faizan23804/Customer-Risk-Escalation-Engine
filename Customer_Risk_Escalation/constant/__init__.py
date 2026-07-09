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
Data Ingestion related constant start with Data_Ingestion VAR name.
"""
DATA_INGESTION_DIR_NAME:str = "data_ingestion"
DATA_INGESTION_DATA_STORE_DIR:str = "data_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"