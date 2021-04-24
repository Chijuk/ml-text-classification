import enum
import json
import logging

log = logging.getLogger("settings")


class SettingType(enum.Enum):
    cleaner = "cleaner"
    trainer = "trainer"


class DataBalancingSetting:
    def __init__(self, min_class_data, under_sampling_value, over_sampling_value) -> None:
        self.min_class_data = min_class_data
        self.under_sampling_value = under_sampling_value
        self.over_sampling_value = over_sampling_value


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


class CleanerSetting:
    def __init__(self, **kwargs) -> None:
        if kwargs['name'] == "": raise ValueError("name is empty")
        self.name = kwargs['name']
        if kwargs['input_data_path'] == "":
            raise ValueError("input_data_path is empty")
        if kwargs['output_data_path'] == "":
            raise ValueError("output_data_path is empty")
        self.input_data_path = kwargs['input_data_path']
        self.output_data_path = kwargs['output_data_path']
        self.log_path = kwargs['log_path']
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


def get_setting(path: str, setting_type: SettingType) -> CleanerSetting:
    """Get settings object deserialized from JSON

    :param path: path to settings file
    :param setting_type: enum SettingType
    :return: CleanerSetting() for SettingType.cleaner
    :raises FileNotFoundError: if path is empty
    """
    if path == "":
        raise FileNotFoundError("Path %s does not exist!" % path)
    with open(path, "r", encoding="utf-8") as file:
        if setting_type == SettingType.cleaner:
            settings = CleanerSetting(**json.load(file))
            # settings.validate()
            return settings
        elif setting_type == SettingType.trainer:
            pass
        else:
            raise TypeError("Unknown setting type %s" % setting_type)
