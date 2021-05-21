import copy
import logging
import os
from pathlib import WindowsPath

import pandas as pd
from flashtext import KeywordProcessor

import settings
import utils.text_utils as text_utils
from utils.file_utils import process_path

log = logging.getLogger("stop_words")
STOP_WORDS_UK = "ukrainian_dict.txt"
STOP_WORDS_RU = "russian_dict.txt"


def load(file_path: str) -> pd.DataFrame:
    """
    Load file and return Dataframe with 'keywords' column

    :param file_path: relative file path
    :return: DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File {} does not exist".format(file_path))
    stopwords = pd.read_csv(file_path, header=None, names=['keywords'], sep="\t")
    log.info("Loaded {} stop words from {}".format(str(len(stopwords)), file_path))
    return stopwords


def load_default(ru: False, uk: False) -> pd.DataFrame:
    """
    Load default list of stop words from the language files
    in '\\resources\\stop_words' path.
    Prepares for flash text (adding clean name '_EMPTY_')

    :param ru: load russian stop words
    :param uk: load ukrainian stop words
    :return: summing DataFrame with stop words. May be empty
    """
    log.info("Loading default stop words...")
    file_path = WindowsPath(os.path.dirname(os.path.abspath(__file__)))
    file_path = str(file_path.parent.parent) + "\\resources\\stop_words\\"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Folder {} does not exist".format(file_path))
    stop_words = pd.DataFrame()
    if uk:
        uk = load(file_path + STOP_WORDS_UK)
        stop_words = pd.concat([uk, stop_words], ignore_index=True)
    if ru:
        ru = load(file_path + STOP_WORDS_RU)
        stop_words = pd.concat([ru, stop_words], ignore_index=True)
    if not stop_words.empty > 0:
        stop_words['clean_name'] = "_EMPTY_"
    return stop_words


def load_custom(setting: settings.PreprocessorSetting) -> pd.DataFrame:
    """
    Load custom list of stop words from all files in the path.
    Stop words must be prepared for flashtext (must contain '=>' delimiter for cleaning).
    Clean summing 'keywords' with general text cleaner

    :param setting: cleaner settings object
    :return: summing DataFrame with stop words. May be empty
    """
    log.info("Loading custom stop words...")
    files_path = process_path(setting.stop_words_settings.custom_stop_words_path)
    log.info(f'Searching custom stop words on path: {files_path}')

    stop_words = pd.DataFrame()

    files = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))]
    if len(files) > 0:
        if len(setting.stop_words_settings.use_file_cleanup) > 0:
            # Modify settings copy
            new_setting = copy.deepcopy(setting)
            new_setting.clean_stop_words = False
            new_setting.clean_html = False
            new_setting.clean_email = False
            new_setting.min_words_count = 0
        for f in files:
            words = load(files_path + "\\" + f)
            if not words.empty:
                words[['keywords', 'clean_name']] = words['keywords'].str.split("=>", expand=True)
            if f in setting.stop_words_settings.use_file_cleanup:
                # General words cleaning in stop words
                words['keywords'] = words['keywords'].apply(text_utils.clean_text_with_setting,
                                                            args=(new_setting, None))
                words = words[words['keywords'] != ""]
            stop_words = pd.concat([words, stop_words], ignore_index=True)
    else:
        log.info("No custom stop words in {}".format(files_path))

    return stop_words


def load_processor(setting: settings.PreprocessorSetting) -> KeywordProcessor:
    """
    Return KeywordProcessor object with different words loaded.
    Words are sorted by len descending and without duplicates

    :param setting: cleaner settings object
    :return: KeywordProcessor. May be empty
    """
    stop_words = pd.DataFrame()
    processor = KeywordProcessor()
    stop_words_settings = setting.stop_words_settings

    if stop_words_settings.custom_stop_words_path:
        custom = load_custom(setting)
        if not custom.empty:
            stop_words = pd.concat([custom, stop_words], ignore_index=True)
    if stop_words_settings.use_ru_stop_words or stop_words_settings.use_uk_stop_words:
        default = load_default(stop_words_settings.use_ru_stop_words, stop_words_settings.use_uk_stop_words)
        if not default.empty > 0:
            stop_words = pd.concat([default, stop_words], ignore_index=True)
    if not stop_words.empty:
        stop_words.drop_duplicates(subset=['keywords'], inplace=True)
        stop_words.reset_index(inplace=True, drop=True)
        # Sort by 'keywords' length
        stop_words = stop_words.reindex((-stop_words['keywords'].str.len()).argsort()).reset_index(drop=True)

        for index, row in stop_words.iterrows():
            processor.add_keyword(row['keywords'], row['clean_name'])
    return processor
