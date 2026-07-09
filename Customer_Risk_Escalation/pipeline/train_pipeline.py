from Customer_Risk_Escalation.components.data_ingestion import DataIngestion
from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *
import sys


class trainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def start_data_ingestion(self):
        try:
            logging.info("Getting the data from PostGre SQL DB")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            artifact = data_ingestion.initialize_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)
        

    

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            
        except Exception as e:
            raise CustomException(e, sys)


