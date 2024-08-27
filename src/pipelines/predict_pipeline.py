import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        

'''
Flask App's POST Method:

In the Flask app, when a POST request is made (e.g., when a user submits data), you create an instance of the PredictPipeline class.
This instance is responsible for handling the transformation of user input data and applying the trained model to make predictions.
PredictPipeline Class:

The PredictPipeline class has two main functions:
    1. Data Transformation Function: This converts the input data from the user into a format (like a Pandas DataFrame) that the model can work with.
    2. Prediction Function: This function does the following:
        a) Unpickles the Model and Preprocessor: The model pickle file contains the best model selected during training along with its learned parameters. The preprocessing pickle file contains the transformation pipeline (e.g., scaling, encoding) that was applied to the data during training.
        b) Applies Preprocessing and Prediction: When you unpickle these files and apply them to the new input data:
            The preprocessing steps (e.g., scaling, encoding) are applied to the user input data using the transform method.
            The best model is then used to make predictions on the preprocessed data using the predict method.
        
Unpickling Process: When you unpickle the files, you're not re-running the entire training or preprocessing code. Instead:
    a) Preprocessing: The transform method of the preprocessor is executed to apply the necessary transformations to the input data.
    b) Prediction: The predict method of the best model is executed to generate predictions based on the preprocessed data.
'''