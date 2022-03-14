from pyexpat import model
from importlib_metadata import method_cache
import pandas as pd
import numpy as np 
import joblib
import sklearn
import numpy as np
import pickle

from flask import Flask,render_template,request,jsonify
app=Flask(__name__)

model=pickle.load(open('lr_model.pkl','rb'))

@app.route('/')


def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])

def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)

    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='Y should be {}'.format(output))
    


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')