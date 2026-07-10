from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str      
    row_count: int          
    column_count: int 

@dataclass
class DataTransformationArtifact:
    X_train_path:str
    X_test_path:str
    y_train_path:str
    y_test_path:str

    text_train_path:str
    text_test_path:str
    
    scaler_path:str