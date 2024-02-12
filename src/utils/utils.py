import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception.exception import CustomException
from src.logger.logging import logging


def save_object(file_path,obj):
     try:
          dir_path=os.path.dirname(file_path)
          os.makedirs(dir_path,exist_ok=True)
          with open(file_path,"wb") as file_obj:
               pickle.dump(obj,file_obj)
               logging.info("save the object")
     except Exception as e:
          raise CustomException(e,sys)
     
def evaluate_model(X_train,y_train,X_test,y_test):
     try:
          pass
     except Exception as e:
          raise CustomException(e,sys) 
     
     
def load_object(file_path):
     try:
          with open(file_path,'rb') as file_obj:
               return pickle.load(file_obj)
     except Exception as e:
          logging.info("Exception is occur in load_object function utils ")
          raise CustomException(e,sys)
     
     