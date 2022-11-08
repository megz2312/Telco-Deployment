import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
model = pickle.load(open('logreg.pkl', 'rb'))
cols=['gender', 'SeniorCitizen', 'Partner','tenure','PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'PaperlessBilling','Churn', 'InternetService_DSL','InternetService_Fiber optic']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 12) 
    
    prediction = model.predict(final_features)
    output = int(prediction[0])
    if output == 1:
        text = "Churn"
    else:
        text = "No Churn"

    return render_template('index.html', prediction_text='Employee Income is {}'.format(text))


if __name__ == "__main__":
    app.run(debug=True)













# import json
# import os
# from flask import Flask,jsonify,request
# from flask_cors import CORS
# from predictor.py import logistic_regression_model

# app = Flask(__name__)
# CORS(app)
# @app.route("/logistic_regression/",methods=['GET'])
# def return_arg():
#     gender = request.args.get('gender')
#     seniorcitizen = request.args.get('seniorcitizen')
#     partner = request.args.get('partner')
#     dependents = request.args.get('dependents')
#     tenure = request.args.get('tenure')
#     phoneservice = request.args.get('pgoneservice')
#     onlinesecurity = request.args.get('onlinesecurity')
#     onlinebackup = request.args.get('onlinebackup')
#     deviceprotection = request.args.get('deviceprotection')
#     techsupport = request.args.get('techsupport')
#     paperlessbilling = request.args.get('paperlessbilling')
#     monthlycharges = request.args.get('monthlycharges')
#     totalcharges = request.args.get('totalcharges')
#     churn = request.args.get('churn')
#     internetservice_dsl = request.args.get('internetservice_dsl')
#     internetservice_fiber_optic = request.args.get('internetservice_fiber optic')
#     contract_month_to_month = request.args.get('contract_month-to-month')
#     contract_one_year = request.args.get('contract_one year')

#     logr = logistic_regression_model.predict(gender,seniorcitizen,partner,dependents,tenure,phoneservice,onlinesecurity,onlinebackup,deviceprotection,techsupport,paperlessbilling,monthlycharges,totalcharges,internetservice_dsl,internetservice_fiber_optic,contract_month_to_month,contract_one_year) 
#     price_dict = {
#                     'model':'logreg',
#                     'churn?': logr,
#                     }
#     return jsonify(price_dict)

# @app.route("/",methods=['GET'])
# def default():
#   return "<h1> Welcome to bitcoin price predictor <h1>"

# if __name__ == "__main__":
#     app.run() 