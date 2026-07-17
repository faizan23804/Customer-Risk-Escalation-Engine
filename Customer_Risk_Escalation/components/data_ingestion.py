import os, sys
from Customer_Risk_Escalation.database.sql_download import SQLClient
from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initialize_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Data ingestion started")

            df = SQLClient.download_data(
                output_path=self.data_ingestion_config.raw_data_path
            )

            artifact = DataIngestionArtifact(
                raw_data_path=self.data_ingestion_config.raw_data_path,
                row_count=df.shape[0],
                column_count=df.shape[1]
            )

            logging.info(f"Data ingestion completed. Shape: {df.shape}")
            logging.info("\n" + "="*75)
            return artifact

        except Exception as e:
            raise CustomException(e, sys)