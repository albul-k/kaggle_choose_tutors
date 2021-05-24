from flask import Flask, request, make_response, jsonify
from flask_restful import Api
from flask_cors import CORS

import os
import yaml
import pandas as pd
import dill

import sys  
sys.path.append('./src/')

PARAMS = yaml.safe_load(open(os.path.join('src', 'params.yaml')))

# Load pipeline
MODEL = None
with open(os.path.join('train', 'pipeline.dill'), 'rb') as file:
    MODEL = dill.load(file)

app = Flask(__name__)
cors = CORS(app, resources={r'*': {'origins': '*'}})
api = Api(app)


@app.route('/predict', methods=['POST'])
def predict():

    if request.is_json:
        
        d: dict = {key: request.json.get(key) for key, value in PARAMS['features']['base']}
        X = pd.DataFrame([list(d.values())], columns=list(d.keys()))
        try:
            y = MODEL.predict(X)[0]
            proba = MODEL.predict_proba(X)[0]
            res = make_response(jsonify({'prediction': y, 'probability': proba, 'success': True}), 200)
            return res
        except Exception as err:
            res = make_response(jsonify({'error': err, 'success': False}), 404)
            return res

    res = make_response(jsonify({'error': 'The request body is not JSON', 'success': False}), 400)
    return res


if __name__ == '__main__':
    app.run()