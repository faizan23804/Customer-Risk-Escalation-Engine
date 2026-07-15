import os,sys
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
import warnings

warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from xgboost import XGBClassifier # type: ignore
from lightgbm import LGBMClassifier # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore

from sklearn.metrics import ( # type: ignore
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    
)

import mlflow # type: ignore
import mlflow.sklearn # type: ignore
import mlflow.xgboost # type: ignore
import mlflow.lightgbm # type: ignore

from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *


class ModelTrainer:

    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise CustomException(e,sys)
        

    def load_data(self):
        try:
            X_train = pd.read_csv(self.data_transformation_artifact.X_train_path)
            X_test = pd.read_csv(self.data_transformation_artifact.X_test_path)
            y_train = pd.read_csv(self.data_transformation_artifact.y_train_path).squeeze()
            y_test = pd.read_csv(self.data_transformation_artifact.y_test_path).squeeze()

            logging.info("Loaded all Train and Test Splits")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def calculate_class_weights(self, y_train):
        try:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()

            #Used by XgBoost
            scale_pos_weights = neg/pos

            classes = np.array([0, 1])
            weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
            #Used by DT, RF, LightGBM
            class_weight_dict = dict(zip(classes, weights))

            return class_weight_dict, scale_pos_weights
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def get_models(self, class_weight_dict, scale_pos_weight):
        try:
            models = {
                "Decision_Tree": DecisionTreeClassifier(
                    max_depth        = 10,
                    min_samples_leaf = 50,
                    class_weight     = class_weight_dict,
                    random_state     = 42
                ),
                "Random_Forest": RandomForestClassifier(
                    n_estimators = 200,
                    max_depth    = 10,
                    class_weight = class_weight_dict,
                    n_jobs       = -1,
                    random_state = 42
                ),
                "XGBoost_Baseline": XGBClassifier(
                    n_estimators      = 100,
                    max_depth         = 6,
                    learning_rate     = 0.1,
                    scale_pos_weight  = scale_pos_weight,
                    random_state      = 42,
                    eval_metric       = 'logloss',
                    use_label_encoder = False
                ),
                "XGBoost_Tuned": XGBClassifier(
                    n_estimators      = 300,
                    max_depth         = 5,
                    learning_rate     = 0.05,
                    min_child_weight  = 3,
                    subsample         = 0.8,
                    colsample_bytree  = 0.8,
                    scale_pos_weight  = scale_pos_weight,
                    random_state      = 42,
                    eval_metric       = 'logloss',
                    use_label_encoder = False
                ),
                "LightGBM": LGBMClassifier(
                    n_estimators  = 300,
                    max_depth     = 5,
                    learning_rate = 0.05,
                    subsample     = 0.8,
                    class_weight  = class_weight_dict,
                    random_state  = 42,
                    n_jobs        = -1,
                    verbose       = -1
                )
            }

            logging.info(f"Defined {len(models)} models for training")
            return models

        except Exception as e:
            raise CustomException(e, sys)
        

    def train_and_evaluate_all(self, models, X_train, X_test, y_train, y_test):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

            results     = []
            trained_models = {}

            for model_name, model in models.items():

                logging.info(f"Training {model_name}...")

                with mlflow.start_run(run_name=model_name):

                    model.fit(X_train, y_train)

                    y_pred      = model.predict(X_test)
                    y_pred_prob = model.predict_proba(X_test)[:, 1]

                    metrics = {
                        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
                        "f1_score"  : round(f1_score(y_test, y_pred, zero_division=0), 4),
                        "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
                        "recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
                        "roc_auc"   : round(roc_auc_score(y_test, y_pred_prob), 4)
                    }

                    print(f"  {model_name} — Evaluation Results")
                    logging.info(f"  {model_name} — Evaluation Results")

                    print(f"  Accuracy  : {metrics['accuracy']}")
                    logging.info(f"  Accuracy  : {metrics['accuracy']}")

                    print(f"  F1 Score  : {metrics['f1_score']}")
                    logging.info(f"  F1 Score  : {metrics['f1_score']}")

                    print(f"  Precision : {metrics['precision']}")
                    logging.info(f"  Precision : {metrics['precision']}")

                    print(f"  Recall    : {metrics['recall']}")
                    logging.info(f"  Recall    : {metrics['recall']}")

                    print(f"  ROC AUC   : {metrics['roc_auc']}")
                    logging.info(f"  ROC AUC   : {metrics['roc_auc']}")

                    print(classification_report(y_test, y_pred, target_names=['Not Escalated', 'Escalated']))
                    logging.info(classification_report(y_test, y_pred, target_names=['Not Escalated', 'Escalated']))

                    mlflow.log_params(model.get_params())

                    mlflow.log_metrics(metrics)

                    mlflow.sklearn.log_model(model, model_name)

                    logging.info(f"{model_name} — Recall: {metrics['recall']} | AUC: {metrics['roc_auc']}")


                results.append({
                    'Model'    : model_name,
                    'Accuracy' : metrics['accuracy'],
                    'F1 Score' : metrics['f1_score'],
                    'Precision': metrics['precision'],
                    'Recall'   : metrics['recall'],
                    'ROC AUC'  : metrics['roc_auc']
                })

                trained_models[model_name] = model

            results_df = pd.DataFrame(results).sort_values(
                'Recall', ascending=False
            ).reset_index(drop=True)


            print("  FINAL MODEL COMPARISON")

            print(results_df.to_string(index=False))

            logging.info("All models trained and evaluated")
            return results_df, trained_models

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def select_best_model(self, results_df, trained_models):
        try:
            results_df = results_df.sort_values(by="Recall", ascending=False)

            best_row        = results_df.iloc[0]
            best_model_name = best_row["Model"]
            best_recall     = best_row["Recall"]

            if best_recall < MODEL_TRAINER_EXPECTED_RECALL:
                raise Exception(
                    f"No model meets recall threshold. "
                    f"Best was {best_model_name} with Recall {best_recall:.4f}. "
                    f"Required: {MODEL_TRAINER_EXPECTED_RECALL}"
                )

            best_model = trained_models[best_model_name]

            logging.info(f"Best model selected : {best_model_name}")
            logging.info(f"Best Recall         : {best_recall}")

            return best_model, best_model_name, best_row

        except Exception as e:
            raise CustomException(e, sys)
        
        
    def save_model(self,best_model):
        try:
            joblib.dump(best_model, self.model_trainer_config.trained_model_path)
            logging.info("Best model saved")

        except Exception as e:
            raise CustomException(e, sys)
        

    def initialize_model_trainer(self):
        try:
            logging.info("Model Trainer Started")

            ##Load Data
            X_train,X_test,y_train,y_test = self.load_data()

            #Calculate Weights
            class_weight_dict, scale_pos_weights = self.calculate_class_weights(y_train=y_train)

            #Load Models
            models = self.get_models(class_weight_dict=class_weight_dict, scale_pos_weight=scale_pos_weights)

            #Training and Evaluation
            results_df, trained_models = self.train_and_evaluate_all(
                models=models,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

            #Selecting the best model
            best_model, best_model_name, best_row = self.select_best_model(results_df=results_df, trained_models=trained_models)

            #Saving Best Model
            os.makedirs(self.model_trainer_config.trained_model_dir,exist_ok=True)
            self.save_model(best_model=best_model)

            #returning Model trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
                model_name=best_model_name,
                recall             = float(best_row['Recall']),
                f1_score           = float(best_row['F1 Score']),
                precision          = float(best_row['Precision']),
                roc_auc            = float(best_row['ROC AUC'])
            )

            logging.info("Model Training Completed")
            logging.info(f"Best Model : {best_model_name}")
            logging.info(f"Recall     : {best_row['Recall']}")
            logging.info(f"ROC AUC    : {best_row['ROC AUC']}")
            logging.info(f"Saved to   : {self.model_trainer_config.trained_model_path}")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
