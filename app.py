from flask import Flask, request, jsonify
import joblib
import pandas as pd
app = Flask(__name__)
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return jsonify({
        'churn_prediction': int(prediction),
        'churn_probability': round(float(probability), 3)
    })


if __name__ == '__main__':
    app.run(debug=True)
