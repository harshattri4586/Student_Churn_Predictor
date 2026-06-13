import os
import sys
import dill

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        logging.error(f"Error saving object at {file_path}: {e}")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():

            ## Hyperparameter tuning with GridSearchCV
            gs = GridSearchCV(
                model,
                params[model_name],
                cv= 3,
                scoring='f1',
                n_jobs=-1    
            )

            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred)
            report[model_name] = test_f1

            logging.info(
                f"{model_name} -> F1: {test_f1:.4f} "
                f"| Best Params: {gs.best_params_}"
            )

            return report
        
    except Exception as e:
        raise CustomException(e,sys)