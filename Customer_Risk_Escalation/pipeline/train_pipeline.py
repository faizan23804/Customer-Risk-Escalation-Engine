from Customer_Risk_Escalation.components.data_ingestion import DataIngestion
from Customer_Risk_Escalation.components.data_validation import DataValidation
from Customer_Risk_Escalation.components.data_transformation import DataTransformation
from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *
import sys


class trainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()


    def start_data_ingestion(self):
        try:
            logging.info("Getting the data from PostGre SQL DB")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initialize_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initialize_data_validation()
            logging.info("Performed and Exited the Data Transformation method")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)

        
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact = data_validation_artifact)
            data_transformation_artifact = data_transformation.initialize_data_transformation()
            logging.info("Performed and Exited the Data Transformation method")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        

    

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact = data_validation_artifact)
            
        except Exception as e:
            raise CustomException(e, sys)


