import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns):
        """
        Creates and returns the preprocessing pipeline
        """
        try:
            logging.info("Creating preprocessing pipeline for numerical columns")

            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in creating preprocessing pipeline: {e}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully")

            target_column = 'Target'

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            numerical_columns = X_train.columns.tolist()

            preprocessor = self.get_data_transformer_object(
                numerical_columns
            )

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            logging.info("Scaling completed")

            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(
                X_train_scaled, y_train
            )

            logging.info(
                f"SMOTE applied. New train shape: "
                f"{X_train_balanced.shape}"
            )

            train_arr = np.c_[
                X_train_balanced,
                np.array(y_train_balanced)
            ]

            test_arr = np.c_[
                X_test_scaled,
                np.array(y_test)
            ]
            logging.info("Data transformation completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor saved as pickle file")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)