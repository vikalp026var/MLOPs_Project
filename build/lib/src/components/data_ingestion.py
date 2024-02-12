import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception.exception import CustomException
from src.logger.logging import logging


@dataclass
class DataIngestionConfig:
     raw_data_path:str=os.path.join("artifacts/raw_data.csv")
     train_data_path:str=os.path.join("artifacts/train_data.csv")
     test_data_path:str=os.path.join("artifacts/test_data.csv")
     

class DataIngestion:
     def __init__(self):
          self.ingestion_config=DataIngestionConfig()
     
     def initate_data_ingestion(self):
          logging.info("Data Ingestion Strating...")
          try:
              data=pd.read_csv("https://raw.githubusercontent.com/sunnysavita10/firstproject/main/artifacts/raw.csv")
              logging.info("Reading a Data Frame")
              os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
              data.to_csv(self.ingestion_config.raw_data_path,index=False)
              logging.info("I have saved the raw dataset in artifact folder")
              logging.info("Here I have performed dataset in artifact folder")
              train_data,test_data=train_test_split(data,test_size=0.20)
              logging.info("train test split completed")
              train_data.to_csv(self.ingestion_config.train_data_path,index=False)
              test_data.to_csv(self.ingestion_config.test_data_path,index=False)
              logging.info("data ingestion part completed ")
              return(
                   self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path
              )
              
               
          except Exception as e:
               logging.info("Exception in Data Ingestion Error Occured")
               raise CustomException(e,sys)
     
     
