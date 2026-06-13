import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)

            return prediction,probability
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    """Maps frontend form inputs to model features"""

    def __init__(self,
                 marital_status: int,
                 application_mode: int,
                 application_order: int,
                 course: int,
                 attendance: int,
                 previous_qualification: int,
                 previous_qualification_grade: float,
                 nationality: int,
                 mothers_qualification: int,
                 fathers_qualification: int,
                 mothers_occupation: int,
                 fathers_occupation: int,
                 admission_grade: float,
                 displaced: int,
                 educational_special_needs: int,
                 debtor: int,
                 tuition_fees_up_to_date: int,
                 gender: int,
                 scholarship_holder: int,
                 age_at_enrollment: int,
                 international: int,
                 cu1_credited: int,
                 cu1_enrolled: int,
                 cu1_evaluations: int,
                 cu1_approved: int,
                 cu1_grade: float,
                 cu1_without_evaluations: int,
                 cu2_credited: int,
                 cu2_enrolled: int,
                 cu2_evaluations: int,
                 cu2_approved: int,
                 cu2_grade: float,
                 cu2_without_evaluations: int,
                 unemployment_rate: float,
                 inflation_rate: float,
                 gdp: float):

        self.marital_status = marital_status
        self.application_mode = application_mode
        self.application_order = application_order
        self.course = course
        self.attendance = attendance
        self.previous_qualification = previous_qualification
        self.previous_qualification_grade = previous_qualification_grade
        self.nationality = nationality
        self.mothers_qualification = mothers_qualification
        self.fathers_qualification = fathers_qualification
        self.mothers_occupation = mothers_occupation
        self.fathers_occupation = fathers_occupation
        self.admission_grade = admission_grade
        self.displaced = displaced
        self.educational_special_needs = educational_special_needs
        self.debtor = debtor
        self.tuition_fees_up_to_date = tuition_fees_up_to_date
        self.gender = gender
        self.scholarship_holder = scholarship_holder
        self.age_at_enrollment = age_at_enrollment
        self.international = international
        self.cu1_credited = cu1_credited
        self.cu1_enrolled = cu1_enrolled
        self.cu1_evaluations = cu1_evaluations
        self.cu1_approved = cu1_approved
        self.cu1_grade = cu1_grade
        self.cu1_without_evaluations = cu1_without_evaluations
        self.cu2_credited = cu2_credited
        self.cu2_enrolled = cu2_enrolled
        self.cu2_evaluations = cu2_evaluations
        self.cu2_approved = cu2_approved
        self.cu2_grade = cu2_grade
        self.cu2_without_evaluations = cu2_without_evaluations
        self.unemployment_rate = unemployment_rate
        self.inflation_rate = inflation_rate
        self.gdp = gdp

    def get_data_as_dataframe(self):
        try:
            data = {
                'Marital status': [self.marital_status],
                'Application mode': [self.application_mode],
                'Application order': [self.application_order],
                'Course': [self.course],
                'Daytime/evening attendance\t': [self.attendance],
                'Previous qualification': [self.previous_qualification],
                'Previous qualification (grade)': [self.previous_qualification_grade],
                'Nacionality': [self.nationality],
                "Mother's qualification": [self.mothers_qualification],
                "Father's qualification": [self.fathers_qualification],
                "Mother's occupation": [self.mothers_occupation],
                "Father's occupation": [self.fathers_occupation],
                'Admission grade': [self.admission_grade],
                'Displaced': [self.displaced],
                'Educational special needs': [self.educational_special_needs],
                'Debtor': [self.debtor],
                'Tuition fees up to date': [self.tuition_fees_up_to_date],
                'Gender': [self.gender],
                'Scholarship holder': [self.scholarship_holder],
                'Age at enrollment': [self.age_at_enrollment],
                'International': [self.international],
                'Curricular units 1st sem (credited)': [self.cu1_credited],
                'Curricular units 1st sem (enrolled)': [self.cu1_enrolled],
                'Curricular units 1st sem (evaluations)': [self.cu1_evaluations],
                'Curricular units 1st sem (approved)': [self.cu1_approved],
                'Curricular units 1st sem (grade)': [self.cu1_grade],
                'Curricular units 1st sem (without evaluations)': [self.cu1_without_evaluations],
                'Curricular units 2nd sem (credited)': [self.cu2_credited],
                'Curricular units 2nd sem (enrolled)': [self.cu2_enrolled],
                'Curricular units 2nd sem (evaluations)': [self.cu2_evaluations],
                'Curricular units 2nd sem (approved)': [self.cu2_approved],
                'Curricular units 2nd sem (grade)': [self.cu2_grade],
                'Curricular units 2nd sem (without evaluations)': [self.cu2_without_evaluations],
                'Unemployment rate': [self.unemployment_rate],
                'Inflation rate': [self.inflation_rate],
                'GDP': [self.gdp]
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
