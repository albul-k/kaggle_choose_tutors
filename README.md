# Project name: ml_choose_tutors_backend

## Used stack

* ML: sklearn, pandas, numpy
* API: REST (Flask)

### Model: GradientBoostingClassifier

### Metric: ROC-AUC

## Link to the Kaggle competition

<https://www.kaggle.com/c/choose-tutors>

## Description

The goal is to predict the probability for a tutor to be a proper one for preparing for the math exam.

## Data

* age [20-70] - tutor's age
* years_of_experience [0-10] - number of years of work experience
* lesson_price [100-4000] - price for lesson
* qualification [1-4] - qualification of tutor
* physics [0|1] - the tutor has experience in physics
* chemistry [0|1] - the tutor has experience in chemistry
* biology [0|1] - the tutor has experience in biology
* english [0|1] - the tutor has experience in english
* geography [0|1] - the tutor has experience in geography
* history [0|1] - the tutor has experience in history
* mean_exam_points [30-100] - the average score of students for a given tutor

## Model used

GradientBoostingClassifier

## How to run

### Pull the Docker image and run it

```bash
#!/bin/bash
docker pull albulk/ml_choose_tutors_backend:latest
docker run -d -p 5000:5000 albulk/ml_choose_tutors_backend
```

## Link to the frontend part

<https://github.com/albul-k/ml_choose_tutors_frontend>
