from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
with open('models/logistic_model.pkl', 'rb') as f:
    log_model = pickle.load(f)
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

@app.route("/")
def home():
    return jsonify({"message=":"Iris ML App is Running"}),200

@app.route("/predict",methods=["POST"])
def predict():
    try:
        data = request.get_json()

        model_type = data.get("model_type")
        features = data.get("features")

        if not model_type or not features:
            return jsonify({"error":"model_type and features are required"}),400
        
        data = np.array(features).reshape(1,-1)
        if model_type == "logreg":
            prediction = log_model.predict(data)[0]
            
        elif model_type == "kmeans":
            prediction = kmeans_model.predict(data)[0]
        else:
            return jsonify({"error":"Invalid model_type."}),400
        return jsonify({
            "model_type": model_type,
            "features": features,
            "prediction": int(prediction)
        })
        
    except Exception as e:
        return jsonify({"error":str(e)}),500
    