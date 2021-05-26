import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from settings import TrainerSetting, get_setting, SettingType
from trainer.data_balancer import trim_minor_classes, over_sample_data, get_sample_ratio
from trainer.models import create_cnn
from trainer.visualization import build_graphs
from utils import logger_utils, text_utils
from utils.file_utils import read_dataset, save_dataset, serialize_dict, save_model

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
log = logging.getLogger("trainer")


def prep_tfidf(text_data, tokenizer: Tokenizer) -> np.ndarray:
    return tokenizer.texts_to_matrix(text_data, mode='tfidf')


def prep_bag_of_words(text_data, tokenizer: Tokenizer, max_sequence_length: int) -> np.ndarray:
    # Turns text into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(text_data)
    return sequence.pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post', truncating='post')


def create_dictionary(text_data, num_words: int) -> Tokenizer:
    log.info("Creating vocabulary")
    if num_words == 0:
        num_words = None
    tokenizer = Tokenizer(num_words=num_words, filters=text_utils.TOKEN_FILTER)
    tokenizer.fit_on_texts(text_data)
    log.info(f'Found {len(tokenizer.word_index) + 1} unique tokens')
    return tokenizer


def get_class_weights(y: np.ndarray) -> list:
    return class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)


def save_classes(data: pd.DataFrame, weights: dict, path: str) -> None:
    classes = pd.DataFrame()
    classes['class_id'] = data['class_id']
    classes['class_name'] = data['class_name']
    classes.drop_duplicates(inplace=True)
    classes.sort_values(by=['class_id'], inplace=True)  # ascending
    if weights is not None:
        if classes.shape[0] != len(weights):
            raise ValueError(f'Different shapes! Classes: {classes.shape[0]}, weights:{len(weights)}')
        classes_dict = pd.DataFrame()
        classes_dict['class_id'] = weights.keys()
        classes_dict['class_weight'] = weights.values()
        classes = classes.merge(classes_dict, on='class_id')
    save_dataset(classes, os.path.join(path, 'classes.csv'))


def execute_trainer(setting: TrainerSetting) -> None:
    data = read_dataset(setting.input_data_path)

    balancer_setting = setting.data_balancing_setting
    if setting.use_data_balancing and balancer_setting.min_class_data > 0:
        data = trim_minor_classes(data, balancer_setting.min_class_data)

    if setting.train_data_size == 0: setting.train_data_size = None
    if setting.validation_data_size == 0: setting.validation_data_size = None
    x_train, x_test_val, y_train, y_test_val = model_selection.train_test_split(data, data['class_id'],
                                                                                train_size=setting.train_data_size,
                                                                                random_state=42)
    x_val, x_test, y_val, y_test = model_selection.train_test_split(x_test_val, y_test_val,
                                                                    train_size=setting.validation_data_size,
                                                                    random_state=42)
    log.info(f'Train/test/validation dataset split: {x_train.shape[0]}, {x_test.shape[0]}, {x_val.shape[0]}')

    if setting.use_data_balancing:
        if balancer_setting.over_sampling_value > 0:
            sample_ratio = get_sample_ratio(y_train, balancer_setting.over_sampling_value, "oversample")
            x_train, y_train = over_sample_data(x_train, y_train, sample_ratio)
        elif len(balancer_setting.over_sampling_ratio) > 0:
            x_train, y_train = over_sample_data(x_train, y_train, balancer_setting.over_sampling_ratio)

    if setting.intermediate_data_path != "":
        save_dataset(x_train, os.path.join(setting.intermediate_data_path, 'train.csv'))
        save_dataset(x_test, os.path.join(setting.intermediate_data_path, 'test.csv'))
        save_dataset(x_val, os.path.join(setting.intermediate_data_path, 'val.csv'))

    class_weights = None
    if setting.use_class_weight:
        class_weights = get_class_weights(y_train.values)
        # Використовується dict(enumeration, class_weights) для підтримки масиву [y] з get_dummies()
        # Використовується dict(class_label, class_weights) якщо значення [y] передаються напряму в модель
        class_weights = dict(enumerate(class_weights))  # as class enumeration

    sample_weights = None
    class_weights_dict = None
    if setting.use_sample_weight:
        class_weights_dict = get_class_weights(y_train.values)
        class_weights_dict = dict(zip(np.unique(y_train.values), class_weights_dict))  # as class labels
        sample_weights = class_weight.compute_sample_weight(class_weight=class_weights_dict, y=y_train.values)

    dictionary = create_dictionary(data['text'].values, setting.dict_num_words)
    serialize_dict(setting.output_data_path, dictionary)

    x_train = prep_bag_of_words(x_train['text'].values, dictionary, setting.max_sequence_length)
    x_test = prep_bag_of_words(x_test['text'].values, dictionary, setting.max_sequence_length)
    x_val = prep_bag_of_words(x_val['text'].values, dictionary, setting.max_sequence_length)

    y_train = pd.get_dummies(y_train.values)
    y_test = pd.get_dummies(y_test.values)
    y_val = pd.get_dummies(y_val.values)

    save_classes(data, class_weights_dict, setting.output_data_path)

    if setting.dict_num_words == 0:
        setting.dict_num_words = len(dictionary.word_index) + 1  # Adding 1 because of reserved 0 index
    # lstm = create_lstm(x_train, y_train, setting.model_name, setting.dict_num_words)
    # lstm = create_embed_model(x_train, y_train, setting.model_name, setting.dict_num_words)
    lstm = create_cnn(x_train, y_train, setting.model_name, setting.dict_num_words)
    lstm.summary(print_fn=log.info)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001, verbose=2)]

    log.info(f'Start training model: "{lstm.name}" with {setting.epochs} epochs')
    history = lstm.fit(x_train, y_train, sample_weight=sample_weights, class_weight=class_weights,
                       epochs=setting.epochs,
                       batch_size=setting.batch_size, validation_data=(x_val, y_val),
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
