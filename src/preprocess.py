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

    log.info("Start text cleaning")
    tqdm.pandas(desc="cleaning text", ncols=100, mininterval=1, unit="row", colour="green")
    data["text"] = data["text"].progress_apply(text_utils.clean_text, args=(setting, stop_words))
    data = data[data['text'] != ""]
    data.reset_index(drop=True)
    log.info("Cleaning finished. Saving to csv...")
    output_data_path = os.getcwd() + setting.output_data_path
    head, tail = os.path.split(output_data_path)
    os.makedirs(head, exist_ok=True)
    data.to_csv(output_data_path, sep=';', encoding='utf-8', index=False)
    log.info(f'Saved to: {output_data_path}')


def __read_data(setting: BaseSetting) -> pd.DataFrame:
    input_data_path = os.getcwd() + setting.input_data_path
    log.debug(f'Reading data from: {input_data_path}')
    data = pd.read_csv(input_data_path, header=0, delimiter=";", encoding='utf-8')
    log.info("Data shape: {}".format(data.shape))
    return data


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
