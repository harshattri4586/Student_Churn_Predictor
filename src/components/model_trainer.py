import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.metrics import f1_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trianed_model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10]
                },
                "Decision Tree": {
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5]
                },
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.5, 1.0]
                }
            }

            model_report:dict = evaluate_models(
                X_train,y_train,X_test,y_test,models,params
            )

            logging.info(f"Model Report: {model_report}")

            best_model_score = max(model_report.values())
            best_model_name = max(
                model_report, key= model_report.get
            )
            best_model = models[best_model_name] 

            if best_model_score <0.6:
                raise CustomException(
                    "No best model found - all below threshold",sys
                )
            
            logging.info(
                f"Best Model: {best_model_name}"
                f"with F1: {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trianed_model_file_path,
                obj=best_model
            )

            y_pred = best_model.predict(X_test)
            final_f1 = f1_score(y_test,y_pred)
            print(classification_report(y_test,y_pred))

            return final_f1
        
        except Exception as e:
            raise CustomException(e, sys)