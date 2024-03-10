import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json

os.environ['FLASK_ENV'] = 'production'

app = Flask(__name__)
CORS(app)

if os.path.isfile("./model.pkl"):
   model = joblib.load("model.pkl")
else:
  raise FileNotFoundError

@app.route('/predict', methods=['POST'])
def post():
  record = json.loads(request.data)
  X = np.array([
    record['general_health'],
    record['checkup'],
    record['exercise'],
    record['skin_cancer'],
    record['other_cancer'],
    record['depression'],
    record['diabetes'],
    record['arthritis'],
    record['sex'],
    record['age_category'],
    record['height_cm'],
    record['weight_kg'],
    record['bmi'],
    record['smoking_history'],
    record['alcohol_consumption'],
    record['fruit_consumption'],
    record['green_vegetables_consumption'],
    record['fried_potato_consumption']
]).reshape(1, -1)
  X_df = pd.DataFrame(data=X, columns=['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category', 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption'])
  _y = round(float(model.predict_proba(X_df)[0][1])*100, 2)
  return {"class": _y}
  
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')