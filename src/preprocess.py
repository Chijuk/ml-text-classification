import logging
import os
from pathlib import WindowsPath
import pandas as pd
from flashtext import KeywordProcessor

import logger_utils
import sys
import settings

log = logging.getLogger("preprocess")
STOP_WORDS_UK = "ukrainian.txt"
STOP_WORDS_RU = "russian.txt"


def data_preprocess(setting: settings.CleanerSetting) -> None:
    """

    :type setting: CleanerSetting
    """
    log.debug("Reading data from: %s" % os.path.abspath(os.fspath(setting.data_path)))
    # data = pd.read_csv(setting.data_path)


def load_stop_words(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError("File {} does not exist".format(file_path))
    stopwords = pd.read_csv(file_path, header=None, names=['keywords'], sep="\t")
    log.info("Loaded {} stop words from {}".format(str(len(stopwords)), file_path))
    return stopwords


def load_default_stop_words(ru: True, uk: True) -> pd.DataFrame:
    log.info("Loading default stop words...")
    file_path = WindowsPath(os.path.dirname(os.path.abspath(__file__)))
    file_path = str(file_path.parent) + "\\resources\\stop_words\\"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Folder {} does not exist".format(file_path))
    stop_words = pd.DataFrame()
    if uk:
        uk = load_stop_words(file_path + STOP_WORDS_UK)
        stop_words = pd.concat([uk, stop_words], ignore_index=True)
    if ru:
        ru = load_stop_words(file_path + STOP_WORDS_RU)
        stop_words = pd.concat([ru, stop_words], ignore_index=True)
    return stop_words


def load_custom_stop_words(files_path: str) -> pd.DataFrame:
    log.info("Loading custom stop words...")
    full_path = os.path.abspath(os.fspath(files_path))
    stop_words = pd.DataFrame()

    files = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))]
    if len(files) > 0:
        for f in files:
            words = load_stop_words(full_path + "\\" + f)
            words[['keywords', 'clean_name']] = words['keywords'].str.split("=>", expand=True)
            stop_words = pd.concat([words, stop_words], ignore_index=True)
    else:
        log.info("No custom stop words in {}".format(full_path))
    return stop_words


def get_stop_words_processor(setting: settings.CleanerSetting) -> KeywordProcessor:
    total = pd.DataFrame()
    processor = KeywordProcessor()

    if setting.custom_stop_words_path:
        custom = load_custom_stop_words(setting.custom_stop_words_path)
        total = pd.concat([custom, total], ignore_index=True)
    if setting.use_ru_stop_words or setting.use_uk_stop_words:
        default = load_default_stop_words(setting.use_ru_stop_words, setting.use_uk_stop_words)
        total = pd.concat([default, total], ignore_index=True)
    if not total.empty:
        total.drop_duplicates(subset=['keywords'], inplace=True)
        total.reset_index(inplace=True, drop=True)
        total = total.reindex((-total["keywords"].str.len()).argsort()).reset_index(drop=True)
        for index, row in total.iterrows():
            processor.add_keyword(row["keywords"], row["clean_name"])
    return processor


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide cleaner settings path. Exit...")
    elif len(sys.argv) == 2:
        cleaner_settings = settings.get_setting(str(sys.argv[1]), settings.SettingType.cleaner)
        logger_utils.init_logging(cleaner_settings.log_path + "\\" + cleaner_settings.name)
        log.info("==> Start initialization")
        stop_words_processor = get_stop_words_processor(cleaner_settings)
        log.info("Total stop words: {}".format(len(stop_words_processor)))
        log.info("==> Preprocessor initialized")
        data_preprocess(cleaner_settings)
    else:
        print("Unknown number of arguments! Exit...")
