# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:17:28 2020

@author: Abhik Bhattacharya
"""

import numpy as np
import pickle
from flask import Flask,render_template,jsonify,request

## Initilising the flask application
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

## Routing the application to root folder
@app.route('/')
def home():
    return render_template('index.html')
    
## Rooting the prediction outcome
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_data=[np.array(int_features)]
    prediction = model.predict_proba(final_data)
    output = round(prediction[0][0]*100,2)
    
    return render_template('index.html',prediction_text='Purchasing probability is {}%'.format(output))

if __name__ == '__main__':
    app.run(debug=True)