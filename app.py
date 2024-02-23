from flask import Flask,render_template,request
from flask import Response
import pickle 
import numpy as np
import pandas as pd


application =  Flask(__name__)
app = application

scaler = pickle.load(open('model\StandardScaler.pkl','rb'))
model = pickle.load(open('model\modelforprediction.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/submit',methods=['GET','POST'])
def predict_dib():
    result = ""
    if request.method=='POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = model.predict(new_data)
        
        if predict[0] == 1:
            result = 'diabetes'
        else:
            result= 'Non-diabetes'
        return render_template('single_prediction.html',result=result)
    else:
        return render_template('index.html')

if __name__==('__main__'):
    application.run(port=5000,debug=True)