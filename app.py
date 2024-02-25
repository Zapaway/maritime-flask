from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import io
import numpy as np
import math
import random
import os
from PIL import Image

app = Flask(__name__)
CORS(app)

corresponding_labels = [ 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'glass' ] 

class MultiLogisticRegressionModel():

    def __init__(self, num_features, num_classes, batch = 1, folds = 1, learning_rate = 1e-2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch = batch
        self.folds = folds
        self.weights = [[0 for _ in range(num_features + 1)] for _ in range(num_classes)]

    def get_features(self, x):
        features = []
        features.append(1)
        for (r,g,b,a) in x:
            features.append(r/a)
            features.append(g/a)
            features.append(b/a)
        
        return features        
    
    def get_weights(self):
        return self.weights 
    
    def load_weights(self):
        with open("weights.txt", "r") as f:
            self.weights = eval(f.read())

    def hypothesis(self, x):
        hypotheses = []
        exponents = []
        features = self.get_features(x)

        for k in range(self.num_classes):
            param_vector = self.weights[k]
            exponent = 0

            for param, feature in zip(param_vector, features):
                exponent += param * feature
            
            exponent = math.e ** exponent
            exponents.append(exponent)

        Z = sum(exponents)
        for exponent in exponents:
            hypotheses.append(exponent / Z)
        
        return hypotheses

    def predict(self, x):
        hypotheses = self.hypothesis(x)
        return np.argmax(hypotheses)

    def loss(self, x, y):
        hypothesis = self.hypothesis(x)[y]
        loss = - math.log(hypothesis)
        return loss

    def gradient(self, x, y):
        hypothesis = self.hypothesis(x)
        features = self.get_features(x)
        all_gradients = []

        for k in range(self.num_classes):
            class_gradients = []
            for j in range(len(features)):
                gradient = ((1 if y == k else 0) - hypothesis[k]) * features[j]
                class_gradients.append(-gradient)
            all_gradients.append(class_gradients)
        return all_gradients

def classify(capture_img, pixel_values):
    train_logistic_model = MultiLogisticRegressionModel(77 * 58 * 3, 6)
    train_logistic_model.load_weights()
    # iterations, train_accuracies, test_accuracies = train_logistic_model.train(train_data + test_data)

    index = train_logistic_model.predict(pixel_values)
    return corresponding_labels[index]

@app.route('/capture/<path:file_url>', methods=["GET"])
def process_capture(file_url: str):
    file_url = file_url.replace("captures/", "captures%2F")
    entire_file_url = file_url + "?" + request.query_string.decode()
   
    capture_data = requests.get(entire_file_url).content

    capture_img = Image.open(io.BytesIO(capture_data))
    width, height = capture_img.size
    pixel_values = list(capture_img.getdata())
 
    label = classify(capture_img, pixel_values)
    print(label)
    return jsonify({"hello": "WOAHHHH"})



@app.route('/hello', methods=["GET"])
def hello():
    return jsonify({"hello": "YOOOO"})


# app.run(host="0.0.0.0", port=80)
