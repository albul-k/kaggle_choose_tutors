"""
app.py
"""

import os
import logging
from logging.handlers import RotatingFileHandler

from typing import Tuple
import dill
from flask import Flask, request, make_response, jsonify
from flask_restful import Api
from flask_cors import CORS
import pandas as pd

from src.exceptions import InvalidUsage
from src.response_templates import response_success


# Load pipeline
with open(os.path.join('model', 'pipeline.dill'), 'rb') as file:
    MODEL = dill.load(file)


def predict():
    """Main endpoint

    Returns:
        json: json response
    """

    if not request.is_json or len(request.json) == 0:
        raise InvalidUsage.bad_request()

    df_data = prepare_data(request.json)
    y_pred, y_pred_proba = get_predictions(MODEL, df_data)

    res = make_response(jsonify(response_success(
        predicted_class=y_pred,
        probability=y_pred_proba,
        status_code=200,
    )))
    return res

def prepare_data(request_json: str) -> pd.DataFrame:
    """Prepare data for model

    Parameters
    ----------
    request_json : str
        JSON data

    Returns
    -------
    pd.DataFrame
        DataFrame
    """

    df_data = pd.DataFrame(
        [list(request_json.values())],
        columns=list(request_json.keys())
    )
    return df_data

def get_predictions(model,
                    df_data: pd.DataFrame) -> Tuple[int, float]:
    """Get predictions from model

    Parameters
    ----------
    model : catboost.CatBoostClassifier
        CatBoostClassifier
    df_data : pd.DataFrame
        DataFrame with data from response

    Returns
    -------
    Tuple[int, float]
        (predicted class, probability)
    """

    try:
        y_pred = int(model.predict(df_data)[0])
        y_pred_proba = round(model.predict_proba(df_data)[0][1] * 100, 2)
        return y_pred, y_pred_proba
    except TypeError as err:
        raise InvalidUsage.bad_request() from err

def create_app() -> Flask:
    """Create flask app

    Returns
    -------
    Flask
        Flask app
    """

    app = Flask(__name__)
    CORS(app, resources={r"*": {"origins": "*"}})
    Api(app)
    app.config['SECRET_KEY'] = 'mysecretkey'
    app.add_url_rule('/predict', view_func=predict, methods=['POST'])

    register_errorshandler(app)
    register_logger(app)
    return app

def register_errorshandler(app: Flask) -> None:
    """Regitering errors handler

    Parameters
    ----------
    app : Flask
        Flask app
    """

    def errorhandler(error):
        """Error handler

        Parameters
        ----------
        error : str
            Error

        Returns
        -------
        str
            Error in JSON format
        """

        response = error.to_json()
        response.status_code = error.status_code
        return response

    app.errorhandler(InvalidUsage)(errorhandler)

def register_logger(app: Flask) -> None:
    """Regitering logger

    Parameters
    ----------
    app : Flask
        Flask app
    """

    handler = RotatingFileHandler(
        filename='app.log',
        maxBytes=100000,
        backupCount=10
    )
    handler.setLevel(logging.ERROR)
    app.logger.addHandler(handler)
