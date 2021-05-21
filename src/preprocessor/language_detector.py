import os
from pathlib import WindowsPath

import fasttext
import pymorphy2

current_file_path = WindowsPath(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_LANG_MODEL = str(current_file_path.parent.parent) + "\\resources\\bin\\lid.176.bin"
morph_ru = pymorphy2.MorphAnalyzer(lang="ru")
morph_uk = pymorphy2.MorphAnalyzer(lang="uk")


def detect_language(text):
    """
    Predict language code for text

    :param text: text
    :return: language code in ISO 639
    """
    try:
        labels, probs = LANGUAGE.predict_lang(text)
        if len(labels) > 0:
            return labels[0][len("__label__"):]
        else:
            return None
    except Exception:
        return None


class LanguageIdentification:

    def __init__(self):
        self.model = fasttext.load_model(PRETRAINED_LANG_MODEL)

    def predict_lang(self, text):
        return self.model.predict(text, threshold=0.7)


LANGUAGE = LanguageIdentification()
