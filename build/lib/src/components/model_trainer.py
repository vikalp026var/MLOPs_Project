import pandas as pd 
import numpy as np 

from src.logger.logging import logging
from src.exception.exception import CustomException

import os 
import sys 
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object,evaluate_model
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



@dataclass
class ModelTrainerConfig:
     models={
     'LinearRegression':LinearRegression(),
     'Lasso':Lasso(),
     'Ridge':Ridge(),
     'Elasticnet':ElasticNet(),
     'RandomForest':RandomForestRegressor(),
     'xgboost':XGBRegressor()
}
     r2_list={}
     model_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
     def __init__(self):
          self.model_config=ModelTrainerConfig()
          
     def evaluate_model(self,true,pred):
          r2_scor=r2_score(true,pred)
          mse=mean_squared_error(true,pred)
          mae=mean_absolute_error(true,pred)
          return r2_scor,mse,mae
          
     def initiate_model_trainer(self,train_arr,test_array):
          try:
               X_train=train_arr[:,:-1]
               y_train=train_arr[:,-1]
               X_test=test_array[:,:-1]
               y_test=test_array[:,-1]
               logging.info("X_train,X_test,y_train,y_test is split ")
               models=self.model_config.models 
               for model_name,model in models.items():
                    model.fit(X_train,y_train)
                    logging.info("Model is fit")
                    y_pred=model.predict(X_test)
                    logging.info(f"Y_pred is {y_pred}")
                    print(f"model which is fitted is : {model_name}")
                    r2_scor,mse,mae=self.evaluate_model(y_test,y_pred)
                    # self.model_config.r2_list.append(r2_scor)
                    self.model_config.r2_list[model_name]=r2_scor
                    print(f" MSE of {model_name} is :{mse}")
                    print(f"mae of {model_name} is :{mae}")
                    print(f"r2_score of {model_name} is {r2_scor}")
                    print("="*40)
                    print("\n")
                    logging.info("Model is fit .....")
                    
               return self.model_config.r2_list
          
          except Exception as e:
               logging.info("Exception in initiat model")
               raise CustomException(e,sys)
               
     def best_model(self,train_array,test_array):
          try:
               logging.info("best model calling is start ")
               r2_report=self.initiate_model_trainer(train_array,test_array)
               best=max(sorted(r2_report.values()))
               # logging.info(f"Best model name is {self.model_config.models}")
               for model_name,score in self.model_config.r2_list.items():
                    if score==best:
                         model_nam=model_name
                         print(f"Best model is {model_name} and r2 score is {score}")
                         save_object(
                              file_path=self.model_config.model_path,
                              obj=self.model_config.models[model_nam]
                         )
                         
                         
      
          except Exception as e:
               logging.info("Exception in model training ")
               raise CustomException(e,sys)
     
     
