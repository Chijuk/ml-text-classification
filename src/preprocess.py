import logging
import os
import sys

import pandas as pd
from flashtext import KeywordProcessor
from tqdm.auto import tqdm

import logger_utils
import stop_words_utils
import text_utils
from settings import CleanerSetting, get_setting, SettingType, BaseSetting

log = logging.getLogger("preprocess")


@logger_utils.profile
def data_preprocess(setting: CleanerSetting, stop_words: KeywordProcessor) -> None:
    data = __read_data(setting)

    if setting.drop_all_duplicates or len(setting.drop_duplicates_class_list) > 0:
        data = process_duplicates(data, "class_id", drop_all=setting.drop_all_duplicates,
                                  class_list=setting.drop_duplicates_class_list)
    log.info("Start text cleaning")
    tqdm.pandas(desc="cleaning text", ncols=100, mininterval=1, unit="row", colour="green")
    data["text"] = data["text"].progress_apply(text_utils.clean_text, args=(setting, stop_words))
    log.info(f'Drop {data[data["text"] == ""].shape[0]} row(s) after text cleaning')
    data = data[data['text'] != ""]
    data.reset_index(drop=True)

    log.info(f'Cleaning finished. Data shape: {data.shape}')
    __save_data(data, os.getcwd() + setting.output_data_path)


def process_duplicates(data: pd.DataFrame, group_column: str, drop_all: True, class_list: []) -> pd.DataFrame:
    if drop_all:
        classes = data[group_column].unique()
        log.info(f'Drop duplicates for all dataset within column: {group_column}')
    elif class_list is not None and len(class_list) > 0:
        classes = class_list
        log.info(f'Drop duplicates for {group_column} of: {str(classes)}')
    else:
        log.info(f'Drop duplicates processor skipped...')
        return data
    for class_id in classes:
        duplicates = data[data[group_column] == class_id]['text'].duplicated(keep="first")
        data = data.drop(duplicates[duplicates.values].index)
        log.info(f'=> Drop {duplicates[duplicates.values].shape[0]} duplicate(s) of {group_column}: {class_id}')
    return data


def __read_data(setting: BaseSetting) -> pd.DataFrame:
    input_data_path = os.getcwd() + setting.input_data_path
    log.debug(f'Reading data from: {input_data_path}')
    data = pd.read_csv(input_data_path, header=0, delimiter=";", encoding='utf-8')
    log.info("Data shape: {}".format(data.shape))
    return data


def __save_data(data: pd.DataFrame, path: str) -> None:
    head, tail = os.path.split(path)
    os.makedirs(head, exist_ok=True)
    log.info("Saving to csv...")
    data.to_csv(path, sep=';', encoding='utf-8', index=False)
    log.info(f'Saved to: {path}')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide cleaner settings path. Exit...")
    elif len(sys.argv) == 2:
        cleaner_settings: CleanerSetting = get_setting(str(sys.argv[1]), SettingType.cleaner)
        logger_utils.init_logging(cleaner_settings.log_path + "\\" + cleaner_settings.name)
        log.info("==> Start initialization")
        stop_words_processor = stop_words_utils.load_processor(cleaner_settings)
        log.info("Total stop words: {}".format(len(stop_words_processor)))
        log.info("==> Preprocessor initialized")
        data_preprocess(cleaner_settings, stop_words_processor)
    else:
        print("Unknown number of arguments! Exit...")
