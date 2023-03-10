import pandas as pd 
import numpy as np
import pickle
import json

class heart_disease():
    def __init__(self,data):
        self.data = data

    def loading_files(self):
        with open("artifacts/heart.pkl",'rb') as file:
            self.model = pickle.load(file)

        with open('artifacts/heart_scale.pkl','rb') as file :
            self.scaler = pickle.load(file)

        with open("artifacts/project_data.json",'rb') as file :
            self.project_data = json.load(file)

    def get_heart_disease_prediction(self):
        self.loading_files()

        age = self.data['html_age']
        sex = self.data['html_sex']
        cp = self.data['html_cp']
        trestbps = self.data['html_trestbps']
        chol = self.data['html_chol']
        fbs = self.data['html_fbs']
        restecg = self.data['html_restecg']
        thalach = self.data['html_thalach']
        exang = self.data['html_exang']
        oldpeak = self.data['html_oldpeak']
        slope = self.data['html_slope']
        ca = self.data['html_ca']
        thal = self.data['html_thal']

        user_data = np.zeros(len(self.project_data["columns_name"]))
        user_data[0] = age
        user_data[1] = self.project_data['gender'][sex]
        user_data[2] = cp
        user_data[3] = trestbps
        user_data[4] = chol
        user_data[5] = fbs
        user_data[6] = restecg
        user_data[7] = thalach 
        user_data[8] = exang
        user_data[9] = oldpeak
        user_data[10] = slope
        user_data[11] = ca
        user_data[12] = thal

        
        scaled_data = self.scaler.transform([user_data])
        pred = self.model.predict(scaled_data)
        print(pred)
        return pred

