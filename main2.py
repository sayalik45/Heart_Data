from flask import Flask,render_template,request
import pickle
import json
import numpy as np

with open(r'artifacts/heart_scale.pkl','rb') as file :
    scaler = pickle.load(file)

with open('artifacts/heart.pkl','rb') as file :
    model = pickle.load(file) 

with open("artifacts/project_data.json",'r') as file :
    project_data = json.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data',methods = ['POST'])
def get_data():
    data = request.form

    ## take all inputs from frontend
    age = data['html_age']
    sex = data['html_sex']
    cp = data['html_cp']
    trestbps = data['html_trestbps']
    chol = data['html_chol']
    fbs = data['html_fbs']
    restecg = data['html_restecg']
    thalach = data['html_thalach']
    exang = data['html_exang']
    oldpeak = data['html_oldpeak']
    slope = data['html_slope']
    ca = data['html_ca']
    thal = data['html_thal']

    user_data = np.zeros(len(project_data["columns_name"]))
    user_data[0] = age
    user_data[1] = project_data['gender'][sex]
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

    scale = scaler.fit([user_data])
    scaled_data = scale.transform([user_data])
    pred = model.predict(scaled_data)
    
    return render_template ('index.html',prediction = pred)
    
    
if __name__ == "__main__" :
    app.run(host = "0.0.0.0",port = 5000,debug = True)

  