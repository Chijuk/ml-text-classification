import logging
import os
import pickle
from typing import Any

import pandas as pd

log = logging.getLogger("file_utils")


def process_path(path: str, make_dirs=False, file_extension='') -> str:
    """
    Validate path string.
    If make_dirs=True will make dirs recursively.
    If file_extension is not empty check that path must contain file

    :param path: path
    :param make_dirs: make dirs with path
    :param file_extension: if path contains file in tail
    :return: validated path
    :raises ValueError if path does not exist
    """
    if os.path.isabs(path) and os.path.exists(path):
        return path
    else:
        abs_path = os.path.abspath(os.fspath(path))
        if os.path.exists(abs_path):
            return abs_path
        elif not os.path.exists(abs_path) and make_dirs:
            if file_extension != '':
                head, tail = os.path.split(abs_path)
                if tail[-len(file_extension):] != file_extension:
                    raise ValueError(f'Path does not contains file extension {file_extension}. Path: {abs_path}')
                os.makedirs(head, exist_ok=True)
            else:
                os.makedirs(abs_path, exist_ok=True)
            return abs_path
        else:
            raise ValueError(f'Unknown path: {path}. CWD: {os.getcwd()}')


def read_dataset(file_path: str) -> pd.DataFrame:
    file_path = process_path(file_path)
    log.info(f'Reading data from: {file_path}')
    data = pd.read_csv(file_path, header=0, delimiter=";", encoding='utf-8')
    log.info("Data shape: {}".format(data.shape))
    return data


def save_dataset(data: pd.DataFrame, path: str) -> None:
    path = process_path(path, make_dirs=True, file_extension='.csv')
    log.info("Saving to csv...")
    data.to_csv(path, sep=';', encoding='utf-8', index=False)
    log.info(f'Saved to: {path}')


def serialize_dict(file_path: str, dictionary) -> None:
    path = process_path(file_path, make_dirs=True)
    path = os.path.join(path, 'dict.bin')
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f'Dictionary saved to: {path}')


def deserialize_dict(file_path: str) -> Any:
    log.info(f'Loading dictionary from: {file_path}')
    file_path = process_path(file_path)
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def save_model(file_path: str, model) -> None:
    file_path = process_path(file_path, make_dirs=True)
    file_path = os.path.join(file_path, model.name + '.h5')
    log.info("Saving model...")
    model.save(file_path, save_format='h5')
    log.info(f'Saved to: {file_path}')
