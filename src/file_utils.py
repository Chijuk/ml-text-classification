import logging
import os

import pandas as pd

from settings import BaseSetting

log = logging.getLogger("file_utils")


def read_data(setting: BaseSetting) -> pd.DataFrame:
    input_data_path = os.getcwd() + setting.input_data_path
    log.debug(f'Reading data from: {input_data_path}')
    data = pd.read_csv(input_data_path, header=0, delimiter=";", encoding='utf-8')
    log.info("Data shape: {}".format(data.shape))
    return data


def save_data(data: pd.DataFrame, path: str) -> None:
    # todo: os.getcwd() + path - не зовсім хороша ідея
    head, tail = os.path.split(os.getcwd() + path)
    os.makedirs(head, exist_ok=True)
    log.info("Saving to csv...")
    data.to_csv(os.getcwd() + path, sep=';', encoding='utf-8', index=False)
    log.info(f'Saved to: {os.getcwd() + path}')
