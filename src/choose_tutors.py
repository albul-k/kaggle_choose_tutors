"""
API: Flask app
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from time import strftime
import sys
import dill
from flask import Flask, request, make_response, jsonify
# from flask_restful import Api
# from flask_cors import CORS
import pandas

sys.path.append('./src/')

# Load pipeline
MODEL = None
with open(os.path.join('src', 'train', 'pipeline.dill'), 'rb') as file:
    MODEL = dill.load(file)

HANDLER = RotatingFileHandler(
    filename='app.log', maxBytes=100000, backupCount=10)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(HANDLER)

APP = Flask(__name__)
# CORS = CORS(APP, resources={r'*': {'origins': '*'}})
# api = Api(APP)


@APP.route('/predict', methods=['POST'])
def predict():
    """Main endpoint

    Returns:
        json: json response
    """
    if request.is_json:

        data = pandas.DataFrame(
            [list(request.json.values())],
            columns=list(request.json.keys())
        )
        date = strftime("[%Y-%b-%d %H:%M:%S]")
        LOGGER.debug('%s Data: %s', date, str(request.json))
        try:
            probability = MODEL.predict_proba(data)[0][1]
            res = make_response(
                jsonify({'probability': round(probability*100, 2), 'success': True}), 200)
            return res
        except AttributeError as err:
            LOGGER.debug('%s Exception: %s', date, str(err))
            res = make_response(
                jsonify({'error': str(err), 'success': False}), 404)
            return res

    res = make_response(
        jsonify({'error': 'The request body is not JSON', 'success': False}), 400)
    return res


if __name__ == '__main__':
    APP.run()
