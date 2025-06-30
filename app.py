import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open(r'model.pkl', 'rb'))
le_weather = pickle.load(open(r'encoder_weather.pkl','rb'))
le_holiday = pickle.load(open(r'encoder_holiday.pkl','rb'))
scale = pickle.load(open(r'scaler.pkl','rb'))

@app.route('/')  # Route to display the home page
def home():
    return render_template('index.html')  # Rendering the home page
@app.route('/predict', methods=["POST", "GET"])  # route to show the predictions in a web UI
def predict():
    input_feature = [x for x in request.form.values()]

    # Encode categorical features before scaling
    input_feature[0] = le_holiday.transform([input_feature[0]])[0]
    input_feature[4] = le_weather.transform([input_feature[4]])[0]

    input_feature = [float(x) for x in input_feature]
    
    # Prepare for scaling
    features_values = [np.array(input_feature)]
    scaled_values = scale.transform(features_values)

    # Create DataFrame for model input
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
             'hours', 'minutes', 'seconds']
    data = pandas.DataFrame(scaled_values, columns=names)

    # Predict
    prediction = model.predict(data)
    print(prediction)

    text = "Estimated Traffic Volume is :"
    return render_template("result.html", prediction_text=text + str(prediction))

if __name__ == "__main__":
    # app.run(host = '0.0.0.0', port = 8000. debug = True)
    port = int(os.environ.get('PORT', 5000))
    app.run(port = port, debug=True, use_reloader = False)