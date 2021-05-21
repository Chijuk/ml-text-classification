import json
import logging
import sys

import pandas as pd
from flashtext import KeywordProcessor
from tqdm.auto import tqdm

from settings import PreprocessorSetting, get_setting, SettingType
from utils import logger_utils, text_utils, stop_words_utils
from utils.file_utils import read_dataset, save_dataset

log = logging.getLogger("preprocess")


@logger_utils.profile
def data_preprocess(setting: PreprocessorSetting, stop_words: KeywordProcessor) -> None:
    data = read_dataset(setting.input_data_path)

    if setting.drop_all_duplicates or len(setting.drop_duplicates_class_list) > 0:
        data = process_duplicates(data, "class_id", drop_all=setting.drop_all_duplicates,
                                  class_list=setting.drop_duplicates_class_list)
    log.info("Start text cleaning")
    tqdm.pandas(desc="cleaning text", ncols=100, mininterval=1, unit="row", colour="green")
    data["text"] = data["text"].progress_apply(text_utils.clean_text_with_setting, args=(setting, stop_words))
    log.info(f'Drop {data[data["text"] == ""].shape[0]} row(s) after text cleaning')
    data = data[data['text'] != ""]
    data.reset_index(drop=True)

    log.info(f'Cleaning finished. Data shape: {data.shape}')
    save_dataset(data, setting.output_data_path)


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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide cleaner settings path. Exit...")
    elif len(sys.argv) == 2:
        preprocessor_settings: PreprocessorSetting = get_setting(str(sys.argv[1]), SettingType.cleaner)
        logger_utils.init_logging(preprocessor_settings.log_path + "\\" + preprocessor_settings.name)
        log.info(f'Using settings:\n{json.dumps(preprocessor_settings.__dict__, default=lambda x: x.__dict__)}')
        log.info("==> Start initialization")
        stop_words_processor = stop_words_utils.load_processor(preprocessor_settings)
        log.info("Total stop words: {}".format(len(stop_words_processor)))
        log.info("==> Preprocessor initialized")
        data_preprocess(preprocessor_settings, stop_words_processor)
    else:
        print("Unknown number of arguments! Exit...")
