FROM python:3.7.7-slim-buster
LABEL "repository"="https://github.com/albul-k/ml_choose_tutors_backend"
LABEL "maintainer"="Konstantin Albul"

EXPOSE 5000

# Set the working directory to /app
WORKDIR /usr/src/choose_tutors

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install the dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY run_app.py .
COPY model/ model/

# run the command to start uWSGI
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run_app:APP"]