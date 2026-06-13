import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting Training Pipeline")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            logging.info(
                f"Ingestion done. "
                f"Train: {train_path} | Test: {test_path}"
            )

            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )

            logging.info("Transformation done")

            model_trainer = ModelTrainer()
            f1 = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Trainig done. Final F1 Score: {f1}")
            print(f"\n Trainig Complete! Best Model F1 Score: {f1:.4f}")

            return f1
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()