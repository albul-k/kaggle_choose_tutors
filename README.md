# ml_choose_tutors_backend

## Link to Kaggle

https://www.kaggle.com/c/choose-tutors

### Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -p 8181:8181 -v train:/usr/app albul-k/ml_choose_tutors_backend
```