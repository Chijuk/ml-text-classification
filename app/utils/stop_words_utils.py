import logging
import os
from pathlib import WindowsPath

import pandas as pd
from flashtext import KeywordProcessor

from utils.file_utils import process_path

log = logging.getLogger("stop_words")
STOP_WORDS_UK = "ukrainian_dict.txt"
STOP_WORDS_RU = "russian_dict.txt"


class StopWordsCleaner:

    def __init__(self, load_uk=True, load_ru=True, alt_stop_words_file="", custom_path="", use_file_cleanup=None,
                 splitter="=>", clean_name="_EMPTY_", cleanup_function=None, **kwargs) -> None:
        self.load_uk = load_uk
        self.load_ru = load_ru
        self.alt_stop_words_file = alt_stop_words_file
        self.custom_path = custom_path
        if use_file_cleanup is None:
            self.use_file_cleanup = []
        else:
            self.use_file_cleanup = use_file_cleanup
        self.splitter = splitter
        self.clean_name = clean_name
        self.custom_stop_words = pd.DataFrame()
        self.default_stop_words = set()
        self.processor = KeywordProcessor()
        self.cleanup_function = cleanup_function
        self.cleanup_function_kwargs = kwargs

    @staticmethod
    def _load(file_path: str) -> pd.DataFrame:
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

    def _load_default(self) -> None:
        """
        Load default list of stop words from the language files
        in '\\resources\\stop_words' path.
        Prepares for flash text (adding clean name '_EMPTY_')
        """
        file_path = WindowsPath(os.path.dirname(os.path.abspath(__file__)))
        file_path = str(file_path.parent.parent) + "\\resources\\stop_words\\"
        if not os.path.exists(file_path):
            raise FileNotFoundError("Folder {} does not exist".format(file_path))
        if self.load_uk:
            uk = StopWordsCleaner._load(os.path.join(file_path, STOP_WORDS_UK))
            self.default_stop_words.update(set(uk['keywords']))
        if self.load_ru:
            ru = StopWordsCleaner._load(os.path.join(file_path, STOP_WORDS_RU))
            self.default_stop_words.update(set(ru['keywords']))

    def _load_custom(self) -> None:
        """
        Load custom list of stop words from all files in the path.
        Stop words must be prepared for flashtext (must contain '=>' delimiter for cleaning).
        """
        files_path = process_path(self.custom_path)
        log.info(f'Searching custom stop words on path: {files_path}')

        files = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))]
        if len(files) > 0:
            for f in files:
                words = StopWordsCleaner._load(os.path.join(files_path, f))
                words[['keywords', 'clean_name']] = words['keywords'].str.split(self.splitter, expand=True)
                if self.cleanup_function is not None and len(self.use_file_cleanup) > 0 and f in self.use_file_cleanup:
                    words['keywords'] = words['keywords'].apply(self.cleanup_function, **self.cleanup_function_kwargs)
                    words = words[words['keywords'] != ""]
                self.custom_stop_words = pd.concat([words, self.custom_stop_words], ignore_index=True)
        else:
            log.info("No custom stop words in {}".format(files_path))

    def fit_text(self) -> None:
        """
        Return KeywordProcessor object with different words loaded.
        Words are sorted by len descending and without duplicates

        :return: KeywordProcessor. May be empty
        """
        if self.load_uk or self.load_ru:
            self._load_default()
        if self.alt_stop_words_file != "":
            file_path = process_path(self.alt_stop_words_file)
            alt_stop_words_file = StopWordsCleaner._load(file_path)
            self.default_stop_words.update(set(alt_stop_words_file['keywords']))
        if self.custom_path != "":
            self._load_custom()
        if not self.custom_stop_words.empty:
            self.custom_stop_words.drop_duplicates(subset=['keywords'], inplace=True)
            self.custom_stop_words.reset_index(inplace=True, drop=True)
            # Sort by 'keywords' length
            self.custom_stop_words = self.custom_stop_words.reindex((-self.custom_stop_words['keywords'].str.len())
                                                                    .argsort()).reset_index(drop=True)

            for index, row in self.custom_stop_words.iterrows():
                self.processor.add_keyword(row['keywords'], row['clean_name'])

    def clean(self, text: str) -> str:
        if not self.custom_stop_words.empty:
            return self.custom_stop_words.replace_keywords(text.lower()).replace(self.clean_name, "").strip()
