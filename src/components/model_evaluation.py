import os 
import sys 
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import save_object,load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from src.logger.logging import logging
from src.exception.exception import CustomException


class ModelEvaluation:
     def __init__(self):
          logging.info("Evaluation started")
     
     def eval_metrics(self,actual,pred):
          mse=np.sqrt(mean_squared_error(actual,pred))
          mae=mean_absolute_error(actual,pred)
          r2_scor=r2_score(actual,pred)
          logging.info("Evaluation Metrics capture")
          return mse,mae,r2_scor
     
     
     
     def initiate_model_evlauation(self,train_array,test_array):
          try:
               X_test,y_test=(test_array[:,:-1],test_array[:,-1])
               model_path=os.path.join("artifacts","model.pkl")
               model=load_object(model_path)
               # mlflow.set_registry_uri("")
               logging.info("Model register")
               tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
               print(tracking_url_type_store)
               with mlflow.start_run():
                    prediction=model.predict(X_test)
                    (mse,mae,r2_scor)=self.eval_metrics(y_test,prediction)
                    mlflow.log_metric("mse",mse)
                    mlflow.log_metric("mae",mae)
                    mlflow.log_metric("r2_score",r2_scor)
                    
                    if tracking_url_type_store!="file":
                         mlflow.sklearn.log_model(model,"model",registered_model_name='ml_model')
                    else:
                         mlflow.sklearn.load_model(model,"model")
                         
                    
          except Exception as e:
               logging.info(e)
               raise CustomException(e,sys)
          