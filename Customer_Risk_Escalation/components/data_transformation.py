import pandas as pd # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import os,sys 
import joblib # type: ignore

import warnings
warnings.filterwarnings('ignore')

from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *


class DataTransformation:

    def __init__(self,data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
        
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def handle_missing_values(self,df):
        try:
            df['resolution_summary'] = df['resolution_summary'].fillna(UNRESOLVED_FILL)
            df['is_unresolved'] = df['resolution_time_hours'].isnull().astype(int)
            df['resolution_time_hours'] = df['resolution_time_hours'].fillna(SENTINEL_VALUE)

            logging.info("Replaced the Null rows with appropriate values.")
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def target_column(self,df):
        try:
            cond_sentiment = df['customer_sentiment'].isin(['negative', 'very_negative'])
            cond_csat      = df['csat_score'] <= 2
            cond_status    = df['status'].isin(['closed_no_action', 'open', 'on_hold'])

            df['escalated'] = (
                cond_sentiment & 
                cond_csat      & 
                cond_status
                ).astype(int)
            
            logging.info("Created Escalated[Target] column using customer_sentiment, csat_score & status columns.")
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def datetime_features(self,df):
        try:
            df['ticket_created_date'] = pd.to_datetime(df['created_at']).dt.normalize()
            df['created_year'] = df['ticket_created_date'].dt.year
            df['created_month'] = df['ticket_created_date'].dt.month
            df['weekday_num'] = df['ticket_created_date'].dt.dayofweek

            logging.info("Created new engineered datetime columns.")
            return df
        except Exception as e:
            raise CustomException(e,sys)

    
    def text_features(self,df):
        try:
            # Combine all text into one NLP input column
            df['combined_text'] = (
            df['initial_message'].astype(str) + " " +
            df['agent_first_reply'].astype(str) + " " +
            df['resolution_summary'].astype(str)
                )

            #Pull out text separately before any encoding
            text_data = df[['combined_text']].copy()

            logging.info("Combines all the text features into separate dataset column.")
            return df, text_data
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def drop_columns(self,df):
        try:
            col = ['ticket_id','customer_id','channel','region','platform','customer_sentiment',
                   'csat_score','status','created_at','ticket_created_date','initial_message',
                    'agent_first_reply','resolution_summary','combined_text']
            df = df.drop(col,axis=1)

            logging.info("Dropped all the unnecessary columns")
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def encode_cat_cols(self, df):
        try:
            priority_map = {
                'low': 1, 'medium': 2, 'high': 3, 'urgent': 4
            }
            df['priority'] = df['priority'].map(priority_map)

            label_encoders = {}
            cat_cols = ['customer_segment', 'product_area', 
                        'issue_type', 'sla_plan', 'created_month']

            for col in cat_cols:
                le = LabelEncoder()          # fresh encoder per column
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le     # store it

            os.makedirs(self.data_transformation_config.label_encoder_dir,exist_ok=True)

            joblib.dump(label_encoders,self.data_transformation_config.label_encoder_path)

            logging.info(f"Label encoders saved: {list(label_encoders.keys())}")

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def split_data(self,df,text_data):
        try:
            X = df.drop(columns=['escalated'])
            y = df['escalated']
            logging.info("Input Features and Target Vector split completed.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y)
            logging.info("Train & Test Split completed for the Tabular Data.")
            
            text_train = text_data.loc[X_train.index].reset_index(drop=True)
            text_test  = text_data.loc[X_test.index].reset_index(drop=True)
            logging.info("Train & Test Split completed for the Text Data.")
            

            return X_train, X_test, y_train, y_test, text_train,text_test
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def scale_data(self,X_train,X_test):
        try:
            scale_cols = [
                'customer_segment', 'product_area', 'issue_type',
                'sla_plan', 'resolution_time_hours',
                'has_attachment', 'created_year', 'created_month', 'weekday_num',
                'is_unresolved']


            scaler = StandardScaler()
            X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
            X_test[scale_cols] = scaler.transform(X_test[scale_cols])

            logging.info("Scaler Object Created & Saved.")

            os.makedirs(
            os.path.dirname(self.data_transformation_config.scaler_path), 
            exist_ok=True
        )

            joblib.dump(scaler, self.data_transformation_config.scaler_path)
            return X_train, X_test
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initialize_data_transformation(self):
        try:
            df = self.read_data(self.data_ingestion_artifact.raw_data_path)
            logging.info(f"Raw data loaded. Shape: {df.shape}")

            #Handle missing values
            df = self.handle_missing_values(df)

            #Create target column
            df = self.target_column(df)

            #Engineer datetime features
            df = self.datetime_features(df)

            #Create text features
            df, text_data = self.text_features(df)

            #Drop unwanted columns
            df = self.drop_columns(df)

            #Encode categorical columns
            df = self.encode_cat_cols(df)

            #Split data
            X_train, X_test, y_train, y_test, text_train, text_test = self.split_data(df, text_data)

            #Scale features
            X_train, X_test = self.scale_data(X_train, X_test)

            os.makedirs(
                self.data_transformation_config.transformed_train_dir, 
                exist_ok=True
            )
            os.makedirs(
                self.data_transformation_config.transformed_test_dir,  
                exist_ok=True
            )

            #Save Artifacts
            pd.DataFrame(X_train).to_csv(
                self.data_transformation_config.X_train_path, index=False)

            pd.DataFrame(X_test).to_csv(
                self.data_transformation_config.X_test_path, index=False)
            
            y_train.to_csv(
                self.data_transformation_config.y_train_path, index=False)

            y_test.to_csv(
                self.data_transformation_config.y_test_path, index=False)

            text_train.to_csv(
                self.data_transformation_config.text_train_path, index=False)

            text_test.to_csv(
                self.data_transformation_config.text_test_path, index=False)
            
            logging.info("All transformed artifacts saved successfully.")

            #Return Artifacts
            return DataTransformationArtifact(
                X_train_path = self.data_transformation_config.X_train_path,
                X_test_path = self.data_transformation_config.X_test_path,
                y_train_path = self.data_transformation_config.y_train_path,
                y_test_path = self.data_transformation_config.y_test_path,
                text_train_path = self.data_transformation_config.text_train_path,
                text_test_path = self.data_transformation_config.text_test_path,
                scaler_path = self.data_transformation_config.scaler_path,
                label_encoder_path=self.data_transformation_config.label_encoder_path)
        
    
        except Exception as e:
            raise CustomException(e,sys)


