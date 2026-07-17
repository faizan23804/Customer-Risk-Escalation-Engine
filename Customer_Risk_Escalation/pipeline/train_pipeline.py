from Customer_Risk_Escalation.components.data_ingestion import DataIngestion
from Customer_Risk_Escalation.components.data_validation import DataValidation
from Customer_Risk_Escalation.components.data_transformation import DataTransformation
from Customer_Risk_Escalation.components.tabular_model_trainer import ModelTrainer
from Customer_Risk_Escalation.components.nlp_trainer import NLPTrainer
from Customer_Risk_Escalation.components.fusion_model import FusionModel
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
        self.model_trainer_config = ModelTrainerConfig()
        self.nlp_trainer_config = NLPTrainerConfig()
        self.fusion_model_config = FusionConfig()


    def start_data_ingestion(self):
        try:
            logging.info("Start of Pipeline")
            logging.info("Getting the data from PostGre SQL DB")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initialize_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("\n" + "="*75)
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initialize_data_validation()
            logging.info("Performed and Exited the Data Validation method")
            logging.info("\n" + "="*75)
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)

        
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("\n" + "="*75)
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact = data_validation_artifact)
            data_transformation_artifact = data_transformation.initialize_data_transformation()
            logging.info("Performed and Exited the Data Transformation method")
            logging.info("\n" + "="*75)
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_model_trainer(self,data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("\n" + "="*75)
            logging.info("Entered the Tabular start_model_trainer method of TrainPipeline class")

            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initialize_model_trainer()
            logging.info("Performed and Exited the Tabular Model trainer method")
            logging.info("\n" + "="*75)

            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_nlp_trainer(self,data_transformation_artifact: DataTransformationArtifact) -> NLPTrainerArtifact:
        try:
            logging.info("\n" + "="*75)
            logging.info("Entered the NLP start_nlp_trainer method of TrainPipeline class")

            nlp_trainer = NLPTrainer(data_transformation_artifact=data_transformation_artifact,
                                         nlp_trainer_config=self.nlp_trainer_config)
            nlp_trainer_artifact = nlp_trainer.initialize_nlp_trainer()
            logging.info("Performed and Exited the NLP trainer method")
            logging.info("\n" + "="*75)

            return nlp_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def start_model_fusion(self,data_transformation_artifact: DataTransformationArtifact,
                  model_trainer_artifact: ModelTrainerArtifact,
                  nlp_trainer_artifact: NLPTrainerArtifact) -> FusionArtifact:
        try:
            logging.info("\n" + "="*75)
            logging.info("Entered the NLP start_model_fusion method of TrainPipeline class")

            fusion_model = FusionModel(data_transformation_artifact=data_transformation_artifact,
                                      model_trainer_artifact=model_trainer_artifact,
                                        nlp_trainer_artifact=nlp_trainer_artifact,
                                        fusion_model_config=self.fusion_model_config)
            fusion_model_artifact = fusion_model.initialize_fusion_model()
            logging.info("Performed and Exited the Model Fusion method")
            logging.info("\n" + "="*75)
            return fusion_model_artifact
        except Exception as e:
            raise CustomException(e, sys)

        


    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact = data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            nlp_trainer_artifact = self.start_nlp_trainer(data_transformation_artifact=data_transformation_artifact)
            fusion_model_artifact = self.start_model_fusion(data_transformation_artifact=data_transformation_artifact,
                                                            model_trainer_artifact=model_trainer_artifact,
                                                            nlp_trainer_artifact=nlp_trainer_artifact)
        except Exception as e:
            raise CustomException(e, sys)


