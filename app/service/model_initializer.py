import logging
from typing import Union

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

from settings import ServiceSetting, PreprocessorSetting
from trainer.train import prep_bag_of_words
from utils import text_utils
from utils.file_utils import deserialize_dict, read_dataset, process_path
from utils.stop_words_utils import StopWordsCleaner
from utils.text_utils import clean_text_with_setting

log = logging.getLogger("model_initializer")


class ServiceParameterPredictor:
    def __init__(self, classes=None, stop_words_cleaner=None, words_dict=None, model=None) -> None:
        self.classes = classes
        self.stop_words_cleaner: StopWordsCleaner = stop_words_cleaner
        self.words_dict = words_dict
        self.model = model
        self.preprocessed_text = ""
        self.text_sequence = None

    def preprocess_text(self, text: str, preprocessor_settings: PreprocessorSetting) -> str:
        return clean_text_with_setting(text, preprocessor_settings, self.stop_words_cleaner.processor,
                                       self.stop_words_cleaner.default_stop_words)

    def _preprocess_prediction(self, prediction: np.ndarray, top_n=5) -> Union[pd.DataFrame, None]:
        if prediction.shape[0] == 0:
            log.info('There are no predictions!')
            return None
        else:
            if prediction.shape[1] != self.classes.shape[0]:
                raise ValueError(
                    f'Different shapes! Predictions: {prediction.shape[1]}, classes{self.classes.shape[0]}')
            result = pd.DataFrame()
            result['class_id'] = self.classes['class_id']
            result['class_name'] = self.classes['class_name']
            result['probability'] = prediction[0] * 100
            return result.sort_values(by=['probability'], ascending=False).head(top_n)

    def get_prediction(self, text: str, top_n_predictions: int) -> Union[pd.DataFrame, None]:
        self.text_sequence = prep_bag_of_words([text], self.words_dict, self.model.input_shape[1])
        return self._preprocess_prediction(self.model.predict(self.text_sequence), top_n=top_n_predictions)


def load_keras_model(model_path: str) -> Sequential:
    import trainer.models as models
    dependencies = {
        "f1": models.f1
    }
    model = load_model(filepath=model_path, custom_objects=dependencies)
    log.info(f'Model {model.name} loaded!')
    model.summary(print_fn=log.info)
    return model


def load_dictionary(file_path: str) -> Tokenizer:
    words_dict = deserialize_dict(file_path)
    log.info(f'Dictionary loaded (Found {len(words_dict.word_index)} unique tokens)')
    return words_dict


def load_classes(file_path: str) -> pd.DataFrame:
    file_path = process_path(file_path)
    classes = read_dataset(file_path)
    classes = classes[['class_id', 'class_name']]
    log.info(f'Classes loaded: {classes.shape[0]}')
    return classes


def load_stop_words(preprocessor_setting: PreprocessorSetting) -> StopWordsCleaner:
    stop_words_setting = preprocessor_setting.stop_words_settings
    if preprocessor_setting.use_words_lemmatization:
        lemmatize_russian = preprocessor_setting.words_lemmatization_setting.russian
        lemmatize_ukrainian = preprocessor_setting.words_lemmatization_setting.ukrainian
    else:
        lemmatize_russian = False
        lemmatize_ukrainian = False
    if not preprocessor_setting.clean_stop_words:
        stop_words_setting.use_uk_stop_words = False
        stop_words_setting.use_ru_stop_words = False
        stop_words_setting.alt_stop_words_file = ""
        stop_words_setting.custom_stop_words_path = ""
        stop_words_setting.use_file_cleanup = None
    cleaner = StopWordsCleaner(load_uk=stop_words_setting.use_uk_stop_words,
                               load_ru=stop_words_setting.use_ru_stop_words,
                               alt_stop_words_file=stop_words_setting.alt_stop_words_file,
                               custom_path=stop_words_setting.custom_stop_words_path,
                               use_file_cleanup=stop_words_setting.use_file_cleanup,
                               cleanup_function=text_utils.clean_text,
                               lemmatize_russian=lemmatize_russian,
                               lemmatize_ukrainian=lemmatize_ukrainian)
    if preprocessor_setting.clean_stop_words:
        cleaner.fit_text()
        log.info("Total stop words: {}".format(len(cleaner.default_stop_words) + len(cleaner.processor)))
    return cleaner


def init_predictor(service_setting: ServiceSetting,
                   preprocessor_setting: PreprocessorSetting) -> ServiceParameterPredictor:
    log.info("==> Service parameter predictor initialization")
    classes = load_classes(service_setting.classes_path)
    stop_words = None
    if preprocessor_setting.clean_stop_words:
        stop_words = load_stop_words(preprocessor_setting)
    word_dict = load_dictionary(service_setting.dict_path)
    model = load_keras_model(service_setting.model_path)
    log.info("==> Service parameter predictor initialized")
    return ServiceParameterPredictor(classes=classes, stop_words_cleaner=stop_words, words_dict=word_dict, model=model)
