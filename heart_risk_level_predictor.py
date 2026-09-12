from flask import Flask,render_template,request
import joblib
import numpy as np

rf_model=joblib.load('heart_risk_prediction_model.sav')
imputer =joblib.load('heart_risk_prediction_imputer.sav')

app=Flask(__name__) 

@app.route('/')
def index():

    return render_template('patient_details.html')

@app.route('/getresults', methods=['POST'])
def getresults():
    if request.method == 'POST':

       name = request.form['name']
       
       male = float(request.form['male'])
       age = float(request.form['age'])
       education = float(request.form['education'])
       currentSmoker = float(request.form['currentSmoker'])
       cigsPerDay = float(request.form['cigsPerDay'])
       BPMeds = float(request.form['BPMeds'])
       prevalentStroke = float(request.form['prevalentStroke'])
       prevalentHyp = float(request.form['prevalentHyp'])
       diabetes = float(request.form['diabetes'])
       totChol = float(request.form['totChol'])
       sysBP = float(request.form['sysBP'])
       diaBP = float(request.form['diaBP'])
       BMI = float(request.form['BMI'])
       heartRate = float(request.form['heartRate'])
       glucose = float(request.form['glucose'])    

    patient_data = [[
            male, age, education, currentSmoker, cigsPerDay, BPMeds, 
            prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, 
            diaBP, BMI, heartRate, glucose
        ]]

    patient_data_cleaned = imputer.transform(patient_data)
    
    prediction = rf_model.predict(patient_data_cleaned)
    final_prediction = int(prediction[0])

    resultDict = {
    "name": name,
    "prediction": final_prediction
}
    return render_template('patient_results.html',results=resultDict)

app.run(debug=True)