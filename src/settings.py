import json
import enum
import logging

log = logging.getLogger("settings")


class SettingType(enum.Enum):
    cleaner = "cleaner"
    trainer = "trainer"


class WordNormalizationSetting:
    def __init__(self, russian, ukrainian):
        self.russian = russian
        self.ukrainian = ukrainian


class EmailSetting:
    def __init__(self, signatures):
        self.signatures = signatures.split(";")


class CleanerSetting:
    def __init__(self, **kwargs):
        if kwargs['name'] == "": raise ValueError("name is empty")
        self.name = kwargs['name']
        if kwargs['data_path'] == "": raise ValueError("data_path is empty")
        self.data_path = kwargs['data_path']
        self.log_path = kwargs['log_path']
        self.min_words_count = kwargs['min_words_count']
        self.max_words_count = kwargs['max_words_count']
        self.min_word_len = kwargs['min_word_len']
        self.max_word_len = kwargs['max_word_len']
        self.min_class_elements = kwargs['min_class_elements']
        self.custom_stop_words_path = kwargs['custom_stop_words_path']
        self.use_uk_stop_words = kwargs['use_uk_stop_words']
        self.use_ru_stop_words = kwargs['use_ru_stop_words']
        self.clean_email = kwargs['clean_email']
        self.email_setting = EmailSetting(**kwargs['email_setting'])
        self.use_words_normalization = kwargs['use_words_normalization']
        self.words_normalization_setting = WordNormalizationSetting(**kwargs['words_normalization_setting'])


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
