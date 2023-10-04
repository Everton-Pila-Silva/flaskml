import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle

application = Flask(__name__)
CORS(application)
model = pickle.load(open("model.pkl", "rb"))
names = pickle.load(open("names.pkl", "rb"))

@application.route("/")
def home():
    return render_template("index.html")

@application.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)
    output = names[pred[0]]
    return render_template("index.html", prediction_text="Iris " + output)

@application.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])
    output = names[pred[0]]
    return jsonify(output)
