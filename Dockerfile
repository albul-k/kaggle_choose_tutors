FROM python:3.7.7-slim-buster

EXPOSE 5000

# Set the working directory to /app
WORKDIR /usr/app

# Install the dependencies
ADD requirements.txt .
RUN pip install -r requirements.txt

# Install pip requirements
RUN python -m pip install -r requirements.txt

COPY . .

# run the command to start uWSGI
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "web.wsgi:app"]