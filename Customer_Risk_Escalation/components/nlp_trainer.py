import os,sys
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
import warnings

warnings.filterwarnings('ignore')

from transformers import DistilBertTokenizer, DistilBertModel # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import mlflow # type: ignore
from tqdm import tqdm
import torch # type: ignore

from Customer_Risk_Escalation.exceptions.exception import CustomException
from Customer_Risk_Escalation.logger.logging import logging
from Customer_Risk_Escalation.entity.artifact_entity import *
from Customer_Risk_Escalation.entity.config_entity import *




class NLPTrainer:

    def __init__(self, data_transformation_artifact: DataTransformationArtifact, nlp_trainer_config: NLPTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.nlp_trainer_config = nlp_trainer_config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # type: ignore

            logging.info(f"GPU  : {torch.cuda.get_device_name(0)}") # type: ignore
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def load_text_data(self):
        try:
            text_train = pd.read_csv(self.data_transformation_artifact.text_train_path).squeeze()
            text_test  = pd.read_csv(self.data_transformation_artifact.text_test_path).squeeze()

            text_train = text_train.fillna('no text available').reset_index(drop=True) # type: ignore
            text_test  = text_test.fillna('no text available').reset_index(drop=True) # type: ignore

            return text_train, text_test
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def load_model(self):
        try:
            MODEL_NAME = DISTILBERT_MODEL_NAME

            logging.info("Loading tokenizer...")
            tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

            logging.info("Loading model...")
            model = DistilBertModel.from_pretrained(MODEL_NAME)
            model = model.to(self.device)
            model.eval()  # Inference mode — no training happening

            logging.info(f"\nModel loaded on : {self.device}")
            logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            return tokenizer, model
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def build_dataset(self,texts,tokenizer,max_length):
        try:
            class TicketTextDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=128):
                    self.texts     = texts.reset_index(drop=True)
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __len__(self):
                    return len(self.texts)

                def __getitem__(self, idx):
                    text = str(self.texts[idx])

                    encoding = self.tokenizer(
                        text,
                        max_length      = self.max_length,
                        padding         = 'max_length',
                        truncation      = True,
                        return_tensors  = 'pt'
                    )

                    return {
                        'input_ids'      : encoding['input_ids'].squeeze(),
                        'attention_mask' : encoding['attention_mask'].squeeze()
                    }
            return TicketTextDataset(texts,tokenizer,max_length)
        except Exception as e:
            raise CustomException(e,sys)
        

    def extract_embeddings(self,texts,tokenizer,model):
        try:
            dataset    = self.build_dataset(texts, tokenizer, max_length=NLP_MAX_LENGTH)
            dataloader = DataLoader(dataset, batch_size=NLP_BATCH_SIZE, 
                                    shuffle=False, num_workers=0)

            all_embeddings = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Extracting embeddings"):

                    input_ids      = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    outputs = model(
                        input_ids      = input_ids,
                        attention_mask = attention_mask
                    )

                    # CLS token = index 0 of last hidden state
                    # Shape: (batch_size, 768)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]

                    # Move back to CPU and convert to numpy
                    all_embeddings.append(cls_embeddings.cpu().numpy())

            return np.vstack(all_embeddings)
        except Exception as e:
            raise CustomException(e,sys)


    def save_embeddings(self,train_embeddings,test_embeddings):
        try:
            os.makedirs(self.nlp_trainer_config.embeddings_dir,exist_ok=True)

            np.save(self.nlp_trainer_config.train_embeddings_path,train_embeddings)
            np.save(self.nlp_trainer_config.test_embeddings_path,test_embeddings)

            loaded = np.load(self.nlp_trainer_config.train_embeddings_path)
            logging.info(f"Train embeddings saved. Shape: {loaded.shape}")
            logging.info(f"Test  embeddings saved. Shape: {test_embeddings.shape}")
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initialize_nlp_trainer(self):
        try:
            logging.info("NLP Trainer Started")

            ##Load Data
            text_train, text_test = self.load_text_data()

            #Load Model
            tokenizer, model = self.load_model()

            #Extract train Embeddings
            train_embeddings = self.extract_embeddings(texts=text_train,tokenizer=tokenizer,model=model)
            logging.info(f"Train embeddings shape: {train_embeddings.shape}")

            #Extract test Embeddings
            test_embeddings = self.extract_embeddings(texts=text_test,tokenizer=tokenizer,model=model)
            logging.info(f"Test embeddings shape: {test_embeddings.shape}")

            #Saving Embeddings.
            self.save_embeddings(train_embeddings=train_embeddings, test_embeddings=test_embeddings)

            #Log to MLFlow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NLP)

            with mlflow.start_run(run_name = "Distilbert_Embeddings"):
                mlflow.log_param("model_name",    DISTILBERT_MODEL_NAME)
                mlflow.log_param("max_length",    NLP_MAX_LENGTH)
                mlflow.log_param("batch_size",    NLP_BATCH_SIZE)
                mlflow.log_param("device",        str(self.device))
                mlflow.log_param("train_samples", len(text_train))
                mlflow.log_param("test_samples",  len(text_test))
                mlflow.log_param("embedding_dim", 768)

            logging.info("MLFlow run Logged")

            #Build and return Nlp_trainer_artifact
            nlp_trainer_artifact = NLPTrainerArtifact(
                    train_embeddings_path = self.nlp_trainer_config.train_embeddings_path,
                    test_embedding_path  = self.nlp_trainer_config.test_embeddings_path,
                    embedding_dimensions = 768,
                    train_samples         = len(text_train),
                    test_samples          = len(text_test)
                )
            
            logging.info("NLP Trainer Completed")
            logging.info(f"Train embeddings : {self.nlp_trainer_config.train_embeddings_path}")
            logging.info(f"Test embeddings  : {self.nlp_trainer_config.test_embeddings_path}")

            return nlp_trainer_artifact

        except Exception as e:
            raise CustomException(e,sys)