import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model_data=pickle.load(open('insurance.pkl','rb'))
model = model_data['model']
scaler = model_data['scaler']

# Encoding maps 
sex_map = {'female': 0,'male': 1}
smoker_map = {'yes': 1, 'no': 0}
region_map = {'southwest': 3, 'southeast': 2, 'northwest': 1, 'northeast': 0}

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = sex_map[request.form['sex'].lower()]
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = smoker_map[request.form['smoker'].lower()]
    region = region_map[request.form['region'].lower()]
    

    input_data = np.array([age,sex, bmi, children, smoker, region])
    scaled = scaler.transform([input_data])
    prediction = model.predict(scaled)[0]

    return render_template('result.html', prediction_text=f"Predicted Insurance Cost: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)

