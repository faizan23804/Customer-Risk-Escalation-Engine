import os
from pathlib import Path


project_name = "Customer_Risk_Escalation"

list_of_files = [

    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/constant/__init__.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/exceptions/exception.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/logging.py",
    f"{project_name}/database/sql_client.py",
    "app.py",
    "data/raw_data",
    "data/processed_data",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "sql_upload.py",
    "main.py",
    ".github/workflows/main.yaml",
    "README.md"
    

]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
    if  (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
    else:
        print(f"File already exists at: {filepath}")


filepath = " "

try:
    if os.path.exists(filepath) and os.path.isfile(filepath):
        os.remove(filepath)
        print("File deleted successfully")
    else:
        print("Nothing to delete")
except Exception as e:
    print("Safe exit:", e)
    