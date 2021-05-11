import logging
import os
import pickle
from typing import Any

import pandas as pd

log = logging.getLogger("file_utils")


def read_dataset(file_path: str) -> pd.DataFrame:
    log.debug(f'Reading data from: {file_path}')
    data = pd.read_csv(file_path, header=0, delimiter=";", encoding='utf-8')
    log.info("Data shape: {}".format(data.shape))
    return data


def save_dataset(data: pd.DataFrame, path: str) -> None:
    # todo: os.getcwd() + path - не зовсім хороша ідея
    # todo: використовувати os.getcwd() якщо на початку '\\'
    head, tail = os.path.split(os.getcwd() + path)
    os.makedirs(head, exist_ok=True)
    log.info("Saving to csv...")
    data.to_csv(os.getcwd() + path, sep=';', encoding='utf-8', index=False)
    log.info(f'Saved to: {os.getcwd() + path}')


def serialize_dict(file_path: str, dictionary) -> None:
    # todo: os.getcwd() + path - не зовсім хороша ідея
    # todo: використовувати os.getcwd() якщо на початку '\\'
    head, tail = os.path.split(os.getcwd() + file_path + "\\dict.bin")
    os.makedirs(head, exist_ok=True)
    with open(os.getcwd() + file_path + "\\dict.bin", 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f'Dictionary saved to: {os.getcwd() + file_path}')


def deserialize_dict(file_path: str) -> Any:
    log.info(f'Loading dictionary from: {file_path}')
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def read_model():
    ...


def save_model(file_path: str, model) -> None:
    # todo: os.getcwd() + path - не зовсім хороша ідея
    # todo: використовувати os.getcwd() якщо на початку '\\'
    full_path = os.getcwd() + file_path
    os.makedirs(full_path, exist_ok=True)
    log.info("Saving model...")
    model.save(full_path + '\\' + model.name + '.h5', save_format='h5')
    log.info(f'Saved to: {full_path}')
