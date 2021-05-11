import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.utils import class_weight

import logger_utils
import models
import text_utils
from data_balancer import trim_minor_classes, over_sample_data, get_sample_ratio
from file_utils import read_dataset, save_dataset, serialize_dict, save_model
from settings import TrainerSetting, get_setting, SettingType
from visualization import build_graphs

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
log = logging.getLogger("trainer")


# todo in future
def prep_tfidf(text_data, tokenizer: Tokenizer) -> np.ndarray:
    return tokenizer.texts_to_matrix(text_data, mode='tfidf')


def prep_bag_of_words(text_data: str, tokenizer: Tokenizer, max_sequence_length: int) -> np.ndarray:
    # Turns text into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(text_data)
    return sequence.pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post', truncating='post')


def create_dictionary(text_data, num_words: int) -> Tokenizer:
    log.info("Creating vocabulary")
    if num_words == 0:
        num_words = None
    tokenizer = Tokenizer(num_words=num_words, filters=text_utils.TOKEN_FILTER)
    tokenizer.fit_on_texts(text_data)
    log.info(f'Found {len(tokenizer.word_index)} unique tokens')
    return tokenizer


def execute_trainer(setting: TrainerSetting) -> None:
    data = read_dataset(setting.input_data_path)

    balancer_setting = setting.data_balancing_setting
    if setting.use_data_balancing and balancer_setting.min_class_data > 0:
        data = trim_minor_classes(data, balancer_setting.min_class_data)

    if setting.train_data_size == 0:
        setting.train_data_size = None
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data, data['class_id'], train_size=setting.train_data_size, random_state=42)
    log.info(f'Train/test dataset split: {x_train.shape[0]}, {x_test.shape[0]}')
    # todo: separate x_val from x_text in new train_test_split

    if setting.use_data_balancing:
        if balancer_setting.over_sampling_value > 0:
            sample_ratio = get_sample_ratio(data["class_id"], balancer_setting.over_sampling_value, "oversample")
            x_train, y_train = over_sample_data(x_train, y_train, sample_ratio)
        elif len(balancer_setting.over_sampling_ratio) > 0:
            x_train, y_train = over_sample_data(x_train, y_train, balancer_setting.over_sampling_ratio)

    if setting.intermediate_data_path != "":
        save_dataset(x_train, setting.intermediate_data_path)
        # todo: save test and validation?

    class_weights = None
    if setting.use_class_weight:
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train.values),
                                                          y=y_train.values)
        # todo: enumerate or class labels?
        class_weights = dict(enumerate(class_weights))

    sample_weights = None
    if setting.use_sample_weight:
        clazz_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train.values),
                                                          y=y_train.values)
        clazz_weights = dict(zip(np.unique(y_train.values), clazz_weights))
        sample_weights = class_weight.compute_sample_weight(class_weight=clazz_weights, y=y_train.values)

    dictionary = create_dictionary(data['text'].values, setting.dict_num_words)
    serialize_dict(setting.output_data_path, dictionary)
    x_train = prep_bag_of_words(x_train['text'].values, dictionary, setting.max_sequence_length)
    x_test = prep_bag_of_words(x_test['text'].values, dictionary, setting.max_sequence_length)

    y_train = pd.get_dummies(y_train.values)  # todo: .values?
    y_test = pd.get_dummies(y_test.values)  # todo: .values?

    if setting.dict_num_words == 0:
        setting.dict_num_words = len(dictionary.word_counts)
    lstm = models.create_lstm(x_train, y_train, setting.model_name, setting.dict_num_words)
    log.info(lstm.summary())

    callbacks = [EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001, verbose=2)]

    log.info(f'Start training model: "{lstm.name}" with {setting.epochs} epochs')
    history = lstm.fit(x_train, y_train, sample_weight=sample_weights, class_weight=class_weights,
                       epochs=setting.epochs,
                       batch_size=setting.batch_size, validation_split=setting.validation_data_size,
                       callbacks=callbacks)
    metrics = lstm.evaluate(x_test, y_test)
    log.info(f'Model evaluation:\n'
             f'Loss: {metrics[0]}\n'
             f'Accuracy: {metrics[1]}\n'
             f'Precision: {metrics[2]}\n'
             f'Recall: {metrics[3]}\n'
             f'Actual epochs: {len(history.epoch)}')

    save_model(setting.output_data_path, lstm)

    build_graphs(history, setting.output_data_path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide trainer settings path. Exit...")
    elif len(sys.argv) == 2:
        trainer_settings: TrainerSetting = get_setting(str(sys.argv[1]), SettingType.trainer)
        logger_utils.init_logging(trainer_settings.log_path + "\\" + trainer_settings.name)
        log.info(f'Using settings:\n{json.dumps(trainer_settings.__dict__, default=lambda x: x.__dict__)}')
        execute_trainer(trainer_settings)
    else:
        print("Unknown number of arguments! Exit...")
