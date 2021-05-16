from os import write
from typing import Any
import pandas as pd
import pickle
import json
import dill
from sklearn.model_selection import train_test_split
from typing import Iterable


def read_csv_data(file_path: str, encoding: str = 'utf-8', delimeter: str = ';') -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        encoding=encoding,
        delimiter=delimeter,
    )
    return df


def load_pkl_data(path: str) -> pd.DataFrame:
    data = None
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pkl_data(data: Any, path: str) -> None:
    with open(path, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    pass


def load_dill(path: str) -> pd.DataFrame:
    data = None
    with open(path, 'rb') as file:
        data = dill.load(file)
    return data


def save_dill(data: Any, path: str) -> None:
    with open(path, "wb") as file:
        dill.dump(data, file)
    pass


def save_json_data(data: Any, path: str) -> None:
    with open(path, 'w') as file:
        json.dump(data, file)
    pass


def save_text_data(data: Any, path: str) -> None:
    with open(path, 'w') as file:
        file.write(data)
    pass


def split_df(df: pd.DataFrame, test_size: float, seed: int, shuffle: bool = True) -> Iterable[pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle,
    )
    return train, test
