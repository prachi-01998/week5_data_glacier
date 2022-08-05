# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:05:58 2022

@author: prach
"""

import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_stars_data.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('work.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('work.html', prediction_text='Star type is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
