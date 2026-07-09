from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_path: str      
    row_count: int          
    column_count: int 