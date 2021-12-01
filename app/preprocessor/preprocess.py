import json
import logging
import sys

import pandas as pd
from flashtext import KeywordProcessor
from tqdm.auto import tqdm

from settings import PreprocessorSetting, get_setting, SettingType
from utils import logger_utils, text_utils
from utils.file_utils import read_dataset, save_dataset
from utils.stop_words_utils import StopWordsCleaner

log = logging.getLogger("preprocess")


@logger_utils.profile
def data_preprocess(setting: PreprocessorSetting, custom_stop_words: KeywordProcessor, default_stop_words: set) -> None:
    data = read_dataset(setting.input_data_path)

    if setting.drop_all_duplicates or len(setting.drop_duplicates_class_list) > 0:
        data = process_duplicates(data, "class_id", drop_all=setting.drop_all_duplicates,
                                  class_list=setting.drop_duplicates_class_list)
    log.info("Start text cleaning")
    tqdm.pandas(desc="cleaning text", ncols=100, mininterval=1, unit="row", colour="green")
    data["text"] = data["text"].progress_apply(text_utils.clean_text_with_setting,
                                               args=(setting, custom_stop_words, default_stop_words))
    log.info(f'Drop {data[data["text"] == ""].shape[0]} row(s) after text cleaning')
    data = data[data['text'] != ""]
    data.reset_index(drop=True)

    log.info(f'Cleaning finished. Data shape: {data.shape}')
    save_dataset(data, setting.output_data_path)


def process_duplicates(data: pd.DataFrame, group_column: str, drop_all=True, class_list=None) -> pd.DataFrame:
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
        stop_words_setting = preprocessor_settings.stop_words_settings
        cleaner = StopWordsCleaner(load_uk=stop_words_setting.use_uk_stop_words,
                                   load_ru=stop_words_setting.use_ru_stop_words,
                                   alt_stop_words_file=stop_words_setting.alt_stop_words_file,
                                   custom_path=stop_words_setting.custom_stop_words_path,
                                   use_file_cleanup=stop_words_setting.use_file_cleanup,
                                   cleanup_function=text_utils.clean_text,
                                   lemmatize_russian=True, lemmatize_ukrainian=True)
        if preprocessor_settings.clean_stop_words:
            cleaner.fit_text()
            log.info("Total stop words: {}".format(len(cleaner.default_stop_words) + len(cleaner.processor)))
        log.info("==> Preprocessor initialized")
        data_preprocess(setting=preprocessor_settings, custom_stop_words=cleaner.processor,
                        default_stop_words=cleaner.default_stop_words)
    else:
        print("Unknown number of arguments! Exit...")
