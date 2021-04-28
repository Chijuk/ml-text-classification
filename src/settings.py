import enum
import json
import logging
from typing import Union

log = logging.getLogger("settings")


class SettingType(enum.Enum):
    cleaner = "cleaner"
    trainer = "trainer"


class DataBalancingSetting:
    def __init__(self, min_class_data, over_sampling_value, over_sampling_ratio) -> None:
        self.min_class_data = min_class_data
        self.over_sampling_value = over_sampling_value
        self.over_sampling_ratio = over_sampling_ratio


class WordLemmatizationSetting:
    def __init__(self, russian, ukrainian) -> None:
        self.russian = russian
        self.ukrainian = ukrainian


class EmailSetting:
    def __init__(self, signatures) -> None:
        self.signatures = signatures.split(";")


class StopWordsSettings:
    def __init__(self, use_uk_stop_words, use_ru_stop_words, custom_stop_words_path, use_file_cleanup) -> None:
        self.use_uk_stop_words = use_uk_stop_words
        self.use_ru_stop_words = use_ru_stop_words
        self.custom_stop_words_path = custom_stop_words_path
        self.use_file_cleanup = use_file_cleanup


class BaseSetting:
    def __init__(self, **kwargs) -> None:
        if kwargs['name'] == "": raise ValueError("name is empty")
        self.name = kwargs['name']
        if kwargs['input_data_path'] == "": raise ValueError("input_data_path is empty")
        self.input_data_path = kwargs['input_data_path']
        if kwargs['output_data_path'] == "": raise ValueError("output_data_path is empty")
        self.output_data_path = kwargs['output_data_path']
        if kwargs['log_path'] == "": raise ValueError("log_path is empty")
        self.log_path = kwargs['log_path']


class CleanerSetting(BaseSetting):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.drop_all_duplicates = kwargs['drop_all_duplicates']
        self.drop_duplicates_class_list = kwargs['drop_duplicates_class_list']
        self.min_words_count = kwargs['min_words_count']
        self.max_words_count = kwargs['max_words_count']
        self.min_word_len = kwargs['min_word_len']
        self.max_word_len = kwargs['max_word_len']
        self.clean_stop_words = kwargs['clean_stop_words']
        self.stop_words_settings = StopWordsSettings(**kwargs['stop_words_settings'])
        self.clean_email = kwargs['clean_email']
        self.email_setting = EmailSetting(**kwargs['email_setting'])
        self.clean_html = kwargs['clean_html']
        self.use_words_lemmatization = kwargs['use_words_lemmatization']
        self.words_lemmatization_setting = WordLemmatizationSetting(**kwargs['words_lemmatization_setting'])


class TrainerSetting(BaseSetting):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_data_balancing = kwargs['use_data_balancing']
        self.data_balancing_setting = DataBalancingSetting(**kwargs['data_balancing_setting'])


def get_setting(path: str, setting_type: SettingType) -> Union[CleanerSetting, TrainerSetting]:
    """Get settings object deserialized from JSON

    :param path: path to settings file
    :param setting_type: enum SettingType
    :return: class extended from BaseSetting
    :raises FileNotFoundError: if path is empty
    """
    if path == "":
        raise FileNotFoundError("Path %s does not exist!" % path)
    with open(path, "r", encoding="utf-8") as file:
        if setting_type == SettingType.cleaner:
            settings = CleanerSetting(**json.load(file))
            return settings
        elif setting_type == SettingType.trainer:
            settings = TrainerSetting(**json.load(file))
            return settings
        else:
            raise TypeError("Unknown setting type %s" % setting_type)
