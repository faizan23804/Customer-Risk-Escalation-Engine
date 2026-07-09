import os
import sys
from datetime import datetime
from Customer_Risk_Escalation.constant import *
from dataclasses import dataclass


timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    artifact_dir : str = os.path.join(ARTIFACTS_DIR,timestamp)
    timestamp : str = timestamp

training_pipeline_config : TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    raw_data_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_DATA_STORE_DIR, FILE_NAME)