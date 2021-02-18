import json
import enum


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
    def __init__(self, name, data_path, min_words_count, max_words_count, min_word_len, max_word_len,
                 min_class_elements, use_stop_words, stop_words, clean_email, email_setting,
                 use_words_normalization, words_normalization_setting):
        self.name = name,
        self.data_path = data_path
        self.min_words_count = min_words_count
        self.max_words_count = max_words_count
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.min_class_elements = min_class_elements
        self.use_stop_words = use_stop_words
        self.stop_words = stop_words.split(';')
        self.clean_email = clean_email
        self.email_setting = EmailSetting(**email_setting)
        self.use_words_normalization = use_words_normalization
        self.words_normalization_setting = WordNormalizationSetting(**words_normalization_setting)


def get_setting(path: str, setting_type: SettingType) -> object:
    if path == "":
        raise FileNotFoundError("Path %s does not exist!" % path)
    with open(path, "r", encoding="utf-8") as file:
        if setting_type == SettingType.cleaner:
            return CleanerSetting(**json.load(file))
        elif setting_type == SettingType.trainer:
            pass
        else:
            raise TypeError("Unknown setting type %s" % setting_type)
