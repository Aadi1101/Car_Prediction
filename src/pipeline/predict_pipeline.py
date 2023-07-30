import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline():
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(filepath=model_path)
            preprocessor = load_object(filepath=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys) 


class CustomData:
    def __init__(
            self,Year:int,Present_Price:int,Kms_Driven:int,Fuel_Type:str,
            Seller_Type:str,Transmission:str,Owner:int
    ):
        self.Year = Year
        self.Present_Price = Present_Price
        self.Kms_Driven = Kms_Driven
        self.Fuel_Type = Fuel_Type
        self.Seller_Type = Seller_Type
        self.Transmission = Transmission
        self.Owner = Owner

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Year":[self.Year],
                "Present_Price":[self.Present_Price],
                "Kms_Driven":[self.Kms_Driven],
                "Fuel_Type":[self.Fuel_Type],
                "Seller_Type":[self.Seller_Type],
                "Transmission":[self.Transmission],
                "Owner":[self.Owner]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)