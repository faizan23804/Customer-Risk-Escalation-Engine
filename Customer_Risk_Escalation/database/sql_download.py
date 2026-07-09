import os
import sys
import pandas as pd  # type: ignore
from sqlalchemy import create_engine  # type: ignore
from dotenv import load_dotenv               # type: ignore
from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.constant import *
from Customer_Risk_Escalation.entity.config_entity import *

load_dotenv()

class SQLClient:


    @staticmethod
    def get_db_engine():
        try:
            engine = create_engine(os.getenv(SQL_DB_URL_KEY))
            return engine
        except Exception as e:
            raise CustomException(e,sys)
    
        
    @staticmethod
    def download_data(output_path:str):
        try:
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            logging.info("Connecting to database...")
            engine = SQLClient.get_db_engine()

            query = """
                SELECT *
                FROM escalation_dataset
            """
            logging.info("Data Pulling Started")
            df = pd.read_sql(query, engine)

            logging.info(f"Data pulled successfully. Shape: {df.shape}")
            print(f"Data pulled successfully. Shape: {df.shape}")

            df.to_csv(output_path, index=False, encoding='utf-8')
            logging.info(f"Data saved to {output_path}")
            print(f"Data saved to {output_path}")

            return df
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = DataIngestionConfig()
    SQLClient.download_data(output_path=config.raw_data_path)