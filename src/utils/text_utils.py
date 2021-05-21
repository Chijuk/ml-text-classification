import logging
import re

import bs4
from flashtext import KeywordProcessor
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from preprocessor import language_detector as lang
from settings import PreprocessorSetting, WordLemmatizationSetting

log = logging.getLogger("text_utils")
TOKEN_FILTER = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + '’«»'


def lemmatize(word: str, settings: WordLemmatizationSetting):
    language = lang.detect_language(word)
    if language == "ru" and settings.russian:
        p = lang.morph_ru.parse(word)[0]
    elif language == "uk" and settings.ukrainian:
        p = lang.morph_uk.parse(word)[0]
    else:
        return word
    if ((str(p.tag) == 'LATN') | ((str(p.tag) != 'UNKN') & (
            p.tag.POS in ['NOUN', 'VERB', 'ADJF', 'ADJS', 'ADVB', 'INFN', 'PRTF', 'PRTS', 'PRCL', 'GRND']))):
        return p.normal_form
    else:
        return ""


def clean_html_tags(text: str) -> str:
    return bs4.BeautifulSoup(text, "html.parser").get_text(separator=' ')


def clean_url(text: str) -> str:
    return re.sub(r'(http|www)\S+', ' ', str(text))


def clean_email_addresses(text: str) -> str:
    return re.sub(r'[a-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+', ' ', str(text))


def clean_email_signature(text: str, signatures: list) -> str:
    """

    :param text: text
    :param signatures: list of email signatures
    :return: cleaned text
    """
    if signatures[-1].lstrip().rstrip() == "":
        signatures.pop()
    for signature in signatures:
        if text.find(signature.lower()) != -1:
            return text.split(signature.lower(), 1)[0]
    return text


def clean_text_with_setting(text: str, setting: PreprocessorSetting, stop_words: KeywordProcessor) -> str:
    return clean_text(text=text, email_signatures=setting.email_setting.signatures,
                      clean_html=setting.clean_html, clean_email_address=setting.email_setting.clean_address,
                      clean_urls=setting.clean_urls, stop_words=stop_words, min_word_len=setting.min_word_len,
                      max_word_len=setting.max_word_len, lemmatization_setting=setting.words_lemmatization_setting,
                      min_words_count=setting.min_words_count)


def clean_text(text: str, email_signatures="", clean_html=False, clean_email_address=False, clean_urls=False,
               clean_numbers=True, stop_words=None, min_word_len=0, max_word_len=0,
               lemmatization_setting: WordLemmatizationSetting = None,
               min_words_count=0) -> str:
    """
    Execute all cleaner functions

    :return: cleaned text
    """
    if email_signatures != "":
        text = clean_email_signature(text, list(email_signatures))
    if clean_html:
        text = clean_html_tags(text)
    if clean_email_address:
        text = clean_email_addresses(text)
    if clean_urls:
        text = clean_url(text)
    # Очищення пробілів та перенесення стрічок
    text = re.sub(r'^\s+|\n|\r|\s+$', ' ', str(text))
    if clean_numbers:
        text = re.sub(r'[0-9]+', '', str(text))

    # Проблемні знаки для токенізатора
    # text = text.replace("’", "'").replace('"', "").replace("«", "").replace("»", "").replace("''", "")

    # Очищення стоп слів зі списку stop_words
    if stop_words is not None and len(stop_words) > 0:
        text = stop_words.replace_keywords(text.lower()).replace("_EMPTY_", "").strip()

    # Токенізація
    tokens = text_to_word_sequence(text, filters=TOKEN_FILTER)
    # Видалення слів < мінімальної кількості символів
    if min_word_len > 0:
        tokens = [i for i in tokens if (len(i) >= int(min_word_len))]
    # Видалення слів > максимальної кількості символів
    if max_word_len > 0:
        tokens = [i for i in tokens if (len(i) <= int(max_word_len))]
    # Лексикологічна нормалізація слів
    if lemmatization_setting is not None:
        tokens = [lemmatize(i, lemmatization_setting) for i in tokens]

    # Очистити по мінімальній кількості слів
    if len(tokens) < int(min_words_count):
        return ""
    text = "".join([i + " " for i in tokens if (len(i) > 0)])
    text = text.lstrip().rstrip()

    return text
