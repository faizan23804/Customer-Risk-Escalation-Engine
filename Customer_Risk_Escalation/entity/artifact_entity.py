from dataclasses import dataclass
import numpy as np # type: ignore


@dataclass
class DataIngestionArtifact:
    raw_data_path: str      
    row_count: int          
    column_count: int 


@dataclass
class DataValidationArtifact:
    validation_status:bool
    report_file_path:str
    status_file_path:str
    drift_dashboard_file_path: str


@dataclass
class DataTransformationArtifact:
    X_train_path:str
    X_test_path:str
    y_train_path:str
    y_test_path:str

    text_train_path:str
    text_test_path:str
    
    scaler_path:str


@dataclass
class ModelTrainerArtifact:
    trained_model_path:str
    model_name:str
    recall:float
    precision:float
    f1_score:float
    roc_auc:float


@dataclass
class NLPTrainerArtifact:
    train_embeddings_path:str
    test_embedding_path:str
    embedding_dimensions:int
    train_samples: np.array
    test_samples:np.array


@dataclass
class FusionArtifact:
    fusion_model_path:str
    recall:float
    f1_score:float
    roc_auc:float