"""
exceptions.py
"""

from typing import List
from flask import jsonify, Response
from src.response_templates import response_error


BAD_REQUEST = response_error(['The request body is not JSON'], status_code=400)


class InvalidUsage(Exception):
    """Exceptions
    """

    status_code = 500

    def __init__(self, message: List[str], status_code: int = None):
        """Init

        Parameters
        ----------
        message : List[str]
            List of strings with error text
        status_code : [type], optional
            Status code of response, by default None
        """

        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

    def to_json(self) -> Response:
        """JSON serializing with flask jsonify

        Returns
        -------
        cls
            Flask response object
        """

        return jsonify(self.message)

    @classmethod
    def bad_request(cls):
        """Bad request exception

        Returns
        -------
        cls
            Class with exception
        """

        return cls(**BAD_REQUEST)
