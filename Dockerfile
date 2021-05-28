FROM python:3.7.7-slim-buster

EXPOSE 5000

# Set the working directory to /app
WORKDIR /usr/src/choose_tutors

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install the dependencies
ADD requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# run the command to start uWSGI
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server.wsgi:app"]