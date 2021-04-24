import collections
import itertools
import os
import sys

import pandas as pd

from settings import CleanerSetting

SCRIPT_DIR = os.path.dirname(__file__)


def prepare_stop_words(load_path, save_path):
    """
    Може бути корисною. Читаємо файл зі стоп словами,
    додаємо дефолтний "=>_EMPTY_" вкінці для flashtext

    :param load_path: шлях звідки читати файл
    :param save_path: шлях куди записуємо оброблений файл
    """
    settings = CleanerSetting(clean_stop_words=False, clean_html=False, clean_email=False,
                              use_words_normalization=False,
                              min_word_len=0, max_word_len=0, min_words_count=0)

    df = pd.read_csv(load_path, header=None, names=['stop_words'], sep="\t")
    df['stop_words'] = df['stop_words'].map(lambda line: line + "=>_EMPTY_")  # prepare for flashtext
    df.to_csv(save_path, header=None, index=None)


def tokenizer_info(load_path):
    from keras.preprocessing.text import Tokenizer

    data = pd.read_csv(load_path, delimiter=';', encoding='utf-8')
    print("==> Loaded %s records" % str(len(data)))
    data = data[data['text'].notna()]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)  # X is list of lists
    # print(str(X))
    # перетворює дані на 2D Numpy масив і обрізає кількість слів до maxlen. 
    # Якщо слів менше за maxlen - наповнює масив нулями (0) з початку
    # Напрямок "обрізання" і "заповнення" визначається в padding='pre', truncating='pre'
    # X = keras.preprocessing.sequence.pad_sequences(X, maxlen=16) 
    # print(str(X.shape)) # elems, words
    # print(str(tokenizer.index_word)) # list: індекс - унікальне слово
    # print(str(tokenizer.index_docs)) # defaultdict: індекс слова - кількість потраплянь слова в тексті 
    # print(str(tokenizer.word_counts)) # OrderedDict: слово - кількість потраплянь слова в тексті 
    # print(str(tokenizer.word_docs)) # defaultdict: слово - кількість потраплянь слова в тексті     
    print("==> Found %s unique tokens." % len(tokenizer.word_index))
    print("==> TOP 100 words by occurrence:")
    sorted_data = collections.OrderedDict(sorted(tokenizer.word_counts.items(), key=lambda item: item[1], reverse=True))
    for key, value in itertools.islice(sorted_data.items(), 0, 100):
        print(str(value).ljust(5) + " : " + key + "\t")

        # Y = pd.get_dummies(data['class_id'].values) # матриця: х - класи, у - рядки, де клас зустрічається (1 або 0)
    # print(str(Y.shape))
    # print(str(Y))
    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=42)
    # print(str(X_test.shape))
    # print(str(Y_test.shape))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] == "help":
            print("Available parameters:")
            print("prepare stop words dictionary".ljust(42) + " - prepare_stop_words [load_file_path] [save_file_path]")
            print("get general tokenization info on train set".ljust(42) + " - tokenizer_info [load_file_path]")
        elif sys.argv[1] == "prepare_stop_words":
            prepare_stop_words(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "tokenizer_info":
            tokenizer_info(sys.argv[2])
        else:
            print("Unknown parameter!")
    else:
        print("Need more arguments!")
