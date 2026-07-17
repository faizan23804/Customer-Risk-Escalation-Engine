import sys,os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import json

from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *
from Customer_Risk_Escalation.constant import SCHEMA_FILE_PATH
from Customer_Risk_Escalation.utils.main_utils import read_yaml_file, write_yaml_file

from evidently.report import Report # type: ignore
from evidently.metric_preset import DataDriftPreset # type: ignore
from evidently.metrics import DatasetDriftMetric # type: ignore

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="evidently")


class DataValidation:

    def __init__(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e,sys)
        
    
    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    

    def validate_schema(self,df) -> tuple[bool,str]:
        try:
            expected_columns = list(self.schema_config['columns'].keys())
            actual_columns   = df.columns.tolist()
            missing_columns  = [col for col in expected_columns if col not in actual_columns]
            status           = len(missing_columns) == 0
            message = ""

            if not status:
                message = f"Missing columns: {missing_columns}"
                logging.info(message)
            else:
                logging.info("Schema validation passed")
            return status, message
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def validate_data_quality(self,df) -> bool:
        try:
            df_cols = df.columns
            missing_num_cols = []
            missing_cat_cols = []
            missing_target_def_cols = []

            #checks no column is 100% null
            for column in df.columns:
                if df[column].isnull().all():
                    print(f"There are columns with 100% missing values: {column}")
                    logging.info(f"{column} contains all Null values")
                else:
                    print("No column is completely Null")

            #checks numerical columns
            for column in self.schema_config["numerical_columns"]:
                if column not in df_cols:
                    missing_num_cols.append(column)

            if len(missing_num_cols)>0:
                logging.info(f"Missing Numerical Column: {missing_num_cols}")

            #checks categorical columns
            for column in self.schema_config["categorical_columns"]:
                if column not in df_cols:
                    missing_cat_cols.append(column)

            if len(missing_cat_cols)>0:
                logging.info(f"Missing Categorical Column: {missing_cat_cols}")

            #checks target definiing columns
            for column in self.schema_config["target_defining_columns"]:
                if column not in df_cols:
                    missing_target_def_cols.append(column)
            
            if len(missing_target_def_cols)>0:
                logging.info(f"Missing Target Defining Column: {missing_target_def_cols}")
                
            return False if len(missing_cat_cols)>0 or len(missing_num_cols)>0 or len(missing_target_def_cols)>0 else True

        except Exception as e:
            raise CustomException(e,sys)

        
    def detect_data_drift(self, reference_df, current_df) -> bool:
        try:
            numerical_cols = self.schema_config.get("numerical_columns", [])
            numerical_cols = [col for col in numerical_cols
                              if col in reference_df.columns
                              and col in current_df.columns]

            ref_df = reference_df[numerical_cols]
            cur_df = current_df[numerical_cols]

            #Run Evidently drift report
            report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric()
            ])
            report.run(reference_data=ref_df, current_data=cur_df)

            #Create report directory and save HTML
            os.makedirs(
                os.path.dirname(self.data_validation_config.drift_dashboard_file_path),
                exist_ok=True
            )
            report.save_html(self.data_validation_config.drift_dashboard_file_path)
            logging.info(f"Drift report saved to: "
                         f"{self.data_validation_config.drift_dashboard_file_path}")

            #Extract drift result as JSON
            report_dict  = report.as_dict()
            print(json.dumps(report_dict['metrics'][0], indent=2))
            write_yaml_file(
                file_path=self.data_validation_config.report_file_path,
                content=report_dict
            )

            #Check if dataset-level drift detected
            drift_status = report_dict['metrics'][1]['result']['dataset_drift']

            if drift_status:
                logging.warning("Data drift detected — model may need retraining")
            else:
                logging.info("No significant data drift detected")

            return drift_status
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def save_validation_status(self, status: bool, message: str):
        try:
            os.makedirs(
                os.path.dirname(self.data_validation_config.status_file_path),
                exist_ok=True
            )

            result = "PASS" if status else "FAIL"

            with open(self.data_validation_config.status_file_path, 'w') as f:
                f.write(f"Validation Status : {result}\n")
                f.write(f"Message           : {message if message else 'All checks passed'}\n")

            logging.info(f"Validation status saved: {result}")

        except Exception as e:
            raise CustomException(e, sys)


    def initialize_data_validation(self):
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            df = self.read_data(self.data_ingestion_artifact.raw_data_path)

            schema_status, schema_msg = self.validate_schema(df=df)
            if not schema_status:
                validation_error_msg += f"SCHEMA ERROR: {schema_msg} | "
                logging.info(f"Schema validation failed: {schema_msg}")

            quality_status = self.validate_data_quality(df=df)
            if not quality_status:
                validation_error_msg += f"QUALITY ERROR IN DATAFRAME"
                logging.info(f"Data quality validation failed")

            validation_status = len(validation_error_msg) == 0

            if not validation_status:
                logging.info(f"Validation FAILED: {validation_error_msg}")
                self.save_validation_status(
                    status=False,
                    message=validation_error_msg
                )
                #Raise error to stop pipeline
                raise Exception(f"Data validation failed: {validation_error_msg}")
            
            split_idx    = int(len(df) * 0.8)
            reference_df = df.iloc[:split_idx]
            current_df   = df.iloc[split_idx:]

            drift_detected = self.detect_data_drift(
                reference_df=reference_df,
                current_df=current_df
            )

            if drift_detected:
                logging.warning("Data drift detected — pipeline continues "
                                "but model retraining recommended")
                
            
            final_message = "All validations passed"
            if drift_detected:
                final_message += " | WARNING: Data drift detected"

            self.save_validation_status(
                status=True,
                message=final_message)
            
            data_validation_artifact = DataValidationArtifact(
                validation_status         = validation_status,
                report_file_path          = self.data_validation_config.report_file_path,
                drift_dashboard_file_path = self.data_validation_config.drift_dashboard_file_path,
                status_file_path          = self.data_validation_config.status_file_path
            )
            logging.info("Data Validation Completed. Status: PASSED")
            logging.info("\n" + "="*75)
            
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
