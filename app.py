import pickle
from flask import Flask, request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return {"Output":"Hello World"}
    else:
        json_data = request.get_json()
        data = CustomData(
            Year=json_data.get('Year'),
            Present_Price=json_data.get('Present_Price'),
            Kms_Driven=json_data.get('Kms_Driven'),
            Fuel_Type=json_data.get('Fuel_Type'),
            Seller_Type=json_data.get('Seller_Type'),
            Transmission=json_data.get('Transmission'),
            Owner=json_data.get('Owner')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return {"results":results[0]}
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)