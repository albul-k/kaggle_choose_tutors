"""
run_app.py
"""

from src.app import create_app


APP = create_app()


if __name__ == "__main__":
    APP.run()
