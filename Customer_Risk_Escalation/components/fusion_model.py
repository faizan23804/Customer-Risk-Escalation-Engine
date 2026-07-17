import os,sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
import mlflow # type: ignore
import mlflow.lightgbm # type: ignore
import warnings

warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight # type: ignore

from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *


class FusionModel:

    def __init__(self,data_transformation_artifact: DataTransformationArtifact,
                  model_trainer_artifact: ModelTrainerArtifact,
                  nlp_trainer_artifact: NLPTrainerArtifact,
                  fusion_model_config: FusionConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.nlp_trainer_artifact = nlp_trainer_artifact
            self.fusion_model_config = fusion_model_config
        except Exception as e:
            raise CustomException(e,sys)
        

    def load_data(self):
        try:
            X_train = pd.read_csv(self.data_transformation_artifact.X_train_path)
            X_test = pd.read_csv(self.data_transformation_artifact.X_test_path)

            y_train = pd.read_csv(self.data_transformation_artifact.y_train_path).squeeze()
            y_test = pd.read_csv(self.data_transformation_artifact.y_test_path).squeeze()

            train_embeddings = np.load(self.nlp_trainer_artifact.train_embeddings_path)
            test_embeddings = np.load(self.nlp_trainer_artifact.test_embedding_path)

            return X_train,X_test,y_train,y_test,train_embeddings,test_embeddings
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def validate_allignment(self,X_train,X_test,train_embeddings,test_embeddings):
        try:
            if X_train.shape[0] != train_embeddings.shape[0]:
                raise CustomException(
                    f"MISMATCH: X_train {X_train.shape[0]} vs train_embeddings {train_embeddings.shape[0]}", sys
                )
            if X_test.shape[0] != test_embeddings.shape[0]:
                raise CustomException(
                    f"MISMATCH: X_test {X_test.shape[0]} vs test_embeddings {test_embeddings.shape[0]}", sys
                )
            logging.info("Tabular Data and Embeddings are in alignment with each other.")
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def calculate_class_weights(self, y_train):
        try:
            classes = np.array([0, 1])
            weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
            class_weight_dict = dict(zip(classes, weights))

            return class_weight_dict
        except Exception as e:
            raise CustomException(e,sys)
        

    def concatenate_features(self,X_train,X_test,train_embeddings,test_embeddings):
        try:
            #Converting tabular to numpy
            X_train_tab = X_train.values
            X_test_tab  = X_test.values

            #Concatenated along feature axis
            X_train_fused = np.concatenate([X_train_tab, train_embeddings], axis=1)
            X_test_fused  = np.concatenate([X_test_tab,  test_embeddings],  axis=1)

            logging.info(f"  Tabular features   : {X_train_tab.shape[1]}")
            logging.info(f"  Embedding features : {train_embeddings.shape[1]}")

            print(f"  Combined features  : {X_train_fused.shape[1]}")
            logging.info(f"  Combined features  : {X_train_fused.shape[1]}")

            print(f"\n  X_train_fused : {X_train_fused.shape}")
            logging.info(f"\n  X_train_fused : {X_train_fused.shape}")

            print(f"  X_test_fused  : {X_test_fused.shape}")
            logging.info(f"  X_test_fused  : {X_test_fused.shape}")

            return X_train_fused, X_test_fused
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def train_fusion_model(self,X_train_fused, X_test_fused, y_train, y_test,class_weight_dict):
        try:
            fusion_params = {
                "n_estimators"  : 300,
                "max_depth"     : 5,
                "learning_rate" : 0.05,
                "subsample"     : 0.8,
                "class_weight"  : class_weight_dict,
                "random_state"  : 42,
                "n_jobs"        : -1,
                "verbose"       : -1
            }

            mlflow.set_experiment(MLFLOW_EXPERIMENT_FUSION)

            with mlflow.start_run(run_name="LateFusion_LightGBM"):

                #Train 
                fusion_model = LGBMClassifier(**fusion_params)
                fusion_model.fit(
                    X_train_fused, y_train,
                    eval_set=[(X_test_fused, y_test)]
                )

                #Predict
                y_pred      = fusion_model.predict(X_test_fused)
                y_pred_prob = fusion_model.predict_proba(X_test_fused)[:, 1] # type: ignore

                #Metrics
                metrics = {
                    "accuracy"  : round(accuracy_score(y_test, y_pred), 4), # type: ignore
                    "f1_score"  : round(f1_score(y_test, y_pred), 4), # type: ignore
                    "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4), # type: ignore
                    "recall"    : round(recall_score(y_test, y_pred), 4), # type: ignore
                    "roc_auc"   : round(roc_auc_score(y_test, y_pred_prob), 4) # type: ignore
                }

                #Log to MLflow
                mlflow.log_params(fusion_params)
                mlflow.log_metrics(metrics)
                mlflow.lightgbm.log_model(fusion_model, "fusion_lgbm") # type: ignore

                logging.info(f"  Accuracy  : {metrics['accuracy']}")
                print(f"  Accuracy  : {metrics['accuracy']}")

                logging.info(f"  F1 Score  : {metrics['f1_score']}")
                print(f"  F1 Score  : {metrics['f1_score']}")

                logging.info(f"  Precision : {metrics['precision']}")
                print(f"  Precision : {metrics['precision']}")

                logging.info(f"  Recall    : {metrics['recall']}")
                print(f"  Recall    : {metrics['recall']}")
                
                logging.info(f"  ROC AUC   : {metrics['roc_auc']}")
                print(f"  ROC AUC   : {metrics['roc_auc']}")

                logging.info("\nClassification Report:")
                logging.info(classification_report(
                    y_test, y_pred, # type: ignore
                    target_names=['Not Escalated', 'Escalated'],
                    zero_division=0
                ))
                print(classification_report(
                    y_test, y_pred, # type: ignore
                    target_names=['Not Escalated', 'Escalated'],
                    zero_division=0
                ))

                logging.info("MLflow run for Fusion Model logged")
                return fusion_model, metrics
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def save_model(self,fusion_model):
        try:
            os.makedirs(
                os.path.dirname(self.fusion_model_config.fusion_model_path),
                exist_ok=True
            )

            joblib.dump(fusion_model, self.fusion_model_config.fusion_model_path)

            logging.info(f"Fusion model saved: {self.fusion_model_config.fusion_model_path}")
        except Exception as e:
            raise CustomException(e,sys)
    

    def initialize_fusion_model(self):
        try:
            logging.info("Model Fusion Started")

            #load data
            X_train,X_test,y_train,y_test,train_embeddings,test_embeddings = self.load_data()

            #Validate Allignment of tabular data and embeddings
            self.validate_allignment(X_train,X_test,train_embeddings,test_embeddings)

            #Calculate weights
            class_weight_dict = self.calculate_class_weights(y_train=y_train)

            #Concatenate Data
            X_train_fused, X_test_fused = self.concatenate_features(X_train=X_train,X_test=X_test,
                                                                    train_embeddings=train_embeddings,
                                                                    test_embeddings=test_embeddings)
            
            #training the model
            fusion_model, metrics = self.train_fusion_model(X_train_fused=X_train_fused,X_test_fused=X_test_fused,
                                                            y_train=y_train,y_test=y_test,class_weight_dict=class_weight_dict)
            
            #Saving the model object
            self.save_model(fusion_model=fusion_model)

            #Build and return Fusion model artifact
            fusion_model_artifact = FusionArtifact(
                fusion_model_path=self.fusion_model_config.fusion_model_path,
                recall= float(metrics["recall"]),
                f1_score=float(metrics["f1_score"]),
                roc_auc=float(metrics["roc_auc"])
            )
            
            logging.info("Model Fusion Trainer Completed")
            logging.info(f"Fusion Model : {self.fusion_model_config.fusion_model_path}")

            return fusion_model_artifact
        except Exception as e:
            raise CustomException(e,sys)
