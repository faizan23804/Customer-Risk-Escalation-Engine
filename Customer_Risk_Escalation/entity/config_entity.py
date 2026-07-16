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


@dataclass
class DataValidationConfig:
    data_validation_dir:str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    report_dir:str = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_DIR)
    report_file_path:str = os.path.join(report_dir,DATA_VALIDATION_REPORT_FILE_NAME)
    status_file_path:str = os.path.join(data_validation_dir,DATA_VALIDATION_STATUS_FILE)
    drift_dashboard_file_path: str = os.path.join(report_dir,DATA_VALIDATION_DRIFT_DASHBOARD_NAME)


@dataclass
class DataTransformationConfig:
    data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_dir:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRAIN_DIR)
    transformed_test_dir:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TEST_DIR)
    transformed_scaled_dir:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_SCALER_DIR)

    X_train_path:str = os.path.join(transformed_train_dir,X_TRAIN_FILE_NAME)
    X_test_path:str = os.path.join(transformed_test_dir,X_TEST_FILE_NAME)
    y_train_path:str = os.path.join(transformed_train_dir,Y_TRAIN_FILE_NAME)
    y_test_path:str = os.path.join(transformed_test_dir,Y_TEST_FILE_NAME)

    text_train_path:str = os.path.join(transformed_train_dir,TEXT_TRAIN_FILE_NAME)
    text_test_path:str = os.path.join(transformed_test_dir,TEXT_TEST_FILE_NAME)
    
    scaler_path:str = os.path.join(transformed_scaled_dir,SCALER_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    model_trainer_dir:str = os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
    trained_model_dir:str = os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR)
    trained_model_path:str = os.path.join(trained_model_dir,MODEL_TRAINER_MODEL_FILE_NAME)
    expected_recall:float = MODEL_TRAINER_EXPECTED_RECALL
    expected_auc:float = MODEL_TRAINER_EXPECTED_AUC


@dataclass
class NLPTrainerConfig:
    nlp_trainer_dir:str = os.path.join(training_pipeline_config.artifact_dir, NLP_DIR_NAME)
    embeddings_dir:str = os.path.join(nlp_trainer_dir,NLP_EMBEDDINGS_DIR)
    train_embeddings_path:str = os.path.join(embeddings_dir, NLP_TRAIN_EMBEDDINGS_FILE)
    test_embeddings_path:str = os.path.join(embeddings_dir, NLP_TEST_EMBEDDINGS_FILE)
    model_name:str = DISTILBERT_MODEL_NAME
    max_length:int = NLP_MAX_LENGTH
    batch_size:int = NLP_BATCH_SIZE


@dataclass
class FusionConfig:
    fusion_dir:str = os.path.join(training_pipeline_config.artifact_dir,FUSION_DIR_NAME)
    fusion_model_dir:str = os.path.join(fusion_dir,FUSION_MODEL_DIR)
    fusion_model_path:str = os.path.join(fusion_model_dir,FUSION_MODEL_FILE_NAME)
