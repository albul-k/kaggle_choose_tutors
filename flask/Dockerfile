FROM python:3.7.7-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app 
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt

# run the command to start uWSGI
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "web.wsgi:app"]