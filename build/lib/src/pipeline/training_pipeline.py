from src.logger.logging import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# logging.info("this is my perfect ")
dataInge=DataIngestion()
train_path,test_path=dataInge.initate_data_ingestion()
data_transformation=DataTransformation()
train_array,test_array=data_transformation.initialize_data_transformation(train_path,test_path)
model=ModelTrainer()

model.best_model(train_array,test_array)

