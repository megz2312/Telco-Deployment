import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence

app = Flask(__name__)
# model = pickle.load(open('logreg.pkl', 'rb'))
model = pickle.load(open('xgmodel.pkl', 'rb'))
cols=['tenure', 'SeniorCitizen', 'Partner','gender','PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'PaperlessBilling', 'InternetService_DSL', 'InternetService_Fiber optic']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 12) 
    k=str(final_features)
    prediction = model.predict(final_features)
    output = int(prediction[0])
    
    if output == 1:
        text = "\'Churn\'"
    else:
        text = "\'Not Churn\'"
    
    prediction_texts='XGBOOST Model results : Employee is more likely to '+str(text)
#     pred=prediction_texts+'     '+k
#     pred=prediction_text
#     return render_template('index.html', prediction_text='Employee is more likely to {}'.format(text))
    return render_template('index.html', prediction_text=prediction_texts)



if __name__ == "__main__":
    app.run(debug=True)
