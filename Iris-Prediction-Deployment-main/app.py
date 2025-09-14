import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model_data=pickle.load(open('irismodel.pkl','rb'))
model = model_data['model']
scaler = model_data['scaler']

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']
    print(data)
    new_data=scaler.transform([list(data.values())])
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    final_features = scaler.transform([np.array(data)])
    print(data)
    
    output=model.predict(final_features)[0]
    print(output)
    
    return render_template('home.html', prediction_text="Species Name is  {}".format(output))



if __name__=="__main__":
    app.run(debug=True)


