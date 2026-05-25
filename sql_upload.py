import os
import sys
import pandas as pd # type: ignore
from sqlalchemy import create_engine, text   # type: ignore
from dotenv import load_dotenv     # type: ignore       
from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.constant import TABLE_NAME, SQL_DB_URL_KEY



load_dotenv()

class SQLDataUpload:

    def __init__(self):
        try:
            #Read the PostgreSQL URL from system env variable 
            self.db_url = os.getenv(SQL_DB_URL_KEY)

            if self.db_url is None:
                raise Exception(f"Environment key '{SQL_DB_URL_KEY}' is not set.")

            # create_engine() sets up the connection pool 
            # It does NOT open a connection yet. Connection opens only when
            # you actually execute a query (lazy connection).
            self.engine = create_engine(
                self.db_url,
                echo=False   # Set echo=True during debugging to see raw SQL logs
            )

            #Test the connection immediately to see if it works
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))  # Lightweight ping query
            
            logging.info("PostgreSQL Engine created and connection verified.")

        except Exception as e:
            raise CustomException(e, sys)


    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Reads the CSV file and returns DataFrame.
        """
        try:
            
            df = pd.read_csv(file_path)


            df.reset_index(drop=True, inplace=True)

            logging.info(f"CSV loaded successfully. Total records: {len(df)}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


    def insert_data_to_SQL(self, dataframe: pd.DataFrame, table_name: str = TABLE_NAME) -> int:
        """
        Pushes the DataFrame into a PostgreSQL table.
        """
        try:
            dataframe.to_sql(
                name=table_name,       # Table name in PostgreSQL
                con=self.engine,       # SQLAlchemy engine (PostgreSQL connection)
                if_exists='replace',   # Drop table if exists, then recreate
                index=False,           # Don't push DataFrame index as a column
                method='multi'         # Batch insert — much faster than row-by-row
            )

            record_count = len(dataframe)
            logging.info(
                f"Data pushed to PostgreSQL table '{table_name}'. "
                f"Records inserted: {record_count}"
            )
            return record_count

        except Exception as e:
            raise CustomException(e, sys)


#Run this file directly to upload your dataset
if __name__ == '__main__':
    try:
        FILE_PATH = "data/raw_data/customer_tkt.csv"

        uploader = SQLDataUpload()

        # Step 1: Load CSV into DataFrame
        df = uploader.load_csv(file_path=FILE_PATH)
        print(f"Preview of loaded data:\n{df.head()}\n")

        # Step 2: Push DataFrame into PostgreSQL
        count = uploader.insert_data_to_SQL(dataframe=df)
        print(f"Total records inserted into PostgreSQL: {count}")

    except Exception as e:
        raise CustomException(e, sys)