import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)


def prepare_stop_words(load_path, save_path):
    """
    Може бути корисною. Читаємо файл зі стоп словами,
    додаємо дефолтний "=>_EMPTY_" вкінці для flashtext

    :param load_path: шлях звідки читати файл
    :param save_path: шлях куди записуємо оброблений файл
    """

    df = pd.read_csv(load_path, header=None, names=['stop_words'], sep="\t")
    df['stop_words'] = df['stop_words'].map(lambda line: line + "=>_EMPTY_")  # prepare for flashtext
    df.to_csv(save_path, header=None, index=None)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] == "help":
            print("Available parameters:")
            print("prepare stop words dictionary".ljust(42) + " - prepare_stop_words [load_file_path] [save_file_path]")
        elif sys.argv[1] == "prepare_stop_words":
            prepare_stop_words(sys.argv[2], sys.argv[3])
        else:
            print("Unknown parameter!")
    else:
        print("Need more arguments!")
