"""
response_templates.py
"""

from typing import List


def response_success(predicted_class: int,
                     probability: float,
                     status_code: int = 200) -> dict:
    """Template for success response

    Parameters
    ----------
    predicted_class : int
        Predicted class
    probability : float
        Probability of predicted class 1
    status_code : int, optional
        Status code of response, by default 200

    Returns
    -------
    dict
        Dictionary with template
    """

    templ = {
        'message': {
            'predicted_class': predicted_class,
            'probability': probability,
        },
        'status_code': status_code
    }
    return templ

def response_error(data: List[str], status_code: int = 500) -> dict:
    """Template for error response

    Parameters
    ----------
    data : List[str]
        List of strings with error text
    status_code : int, optional
        Status code of response, by default 500

    Returns
    -------
    dict
        Dictionary with template
    """

    templ = {
        'message': {
            'errors': data
        },
        'status_code': status_code
    }
    return templ
