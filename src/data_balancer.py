import logging
from typing import Tuple, Dict

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

import logger_utils
from settings import DataBalancingSetting

log = logging.getLogger("data_balancer")


def execute(data: pd.DataFrame, setting: DataBalancingSetting) -> pd.DataFrame:
    log.info("Start balancing data")
    if setting.min_class_data > 0:
        data = trim_minor_classes(data, setting.min_class_data)
    if setting.over_sampling_value > 0:
        data = over_sample_data(data, data["class_id"], get_sample_ratio(data["class_id"], setting.over_sampling_value,
                                                                         "oversample"))
    elif len(setting.over_sampling_ratio) > 0:
        data = over_sample_data(data, data["class_id"], setting.over_sampling_ratio)

    log.info("Finished balancing data")

    return data


@logger_utils.profile
def trim_minor_classes(data: pd.DataFrame, min_class_data: int) -> pd.DataFrame:
    """
    Delete classes less than min_class_data balancing setting value.

    :param data: data
    :param min_class_data: minimum class data value
    :return: trimmed data
    """
    log.info("Trimming classes shape")
    classes = data['class_id'].unique()

    for class_id in classes:
        class_shape = data[data['class_id'] == class_id].shape[0]
        if class_shape < min_class_data:
            print(f'Deleting under shape class [{class_id}]')
            data = data.drop(data[data['class_id'] == class_id].index)
            data.reset_index(drop=True)
    return data


def over_sample_data(x: pd.DataFrame, y: pd.DataFrame, sample_ratio: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # cast str key to int
    ratio = {}
    for key, value in sample_ratio.items():
        ratio[int(key)] = value
    return RandomOverSampler(sampling_strategy=ratio).fit_resample(x, y)


def get_sample_ratio(y: pd.DataFrame, sample_value: int, strategy: str) -> Dict[int, int]:
    strategies = ["undersample", "oversample"]
    if strategy not in strategies:
        raise ValueError(f'Unknown strategy {strategy}. Possible values: {strategies}')
    if strategy == "undersample":
        raise NotImplementedError("'undersample' strategy not implemented yet")
    if strategy == "oversample":
        classes = y.unique()
        ratio = {}
        for clazz in classes:
            if y[y.values == clazz].count() < sample_value:
                ratio[int(clazz)] = sample_value
        return ratio
