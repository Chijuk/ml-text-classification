import logging
import re

import bs4
from flashtext import KeywordProcessor
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from preprocessor import language_detector as lang
from settings import PreprocessorSetting

log = logging.getLogger("text_utils")
TOKEN_FILTER = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + '’«»'


def lemmatize(word: str, russian=False, ukrainian=False):
    language = lang.detect_language(word)
    if language == "ru" and russian:
        p = lang.morph_ru.parse(word)[0]
    elif language == "uk" and ukrainian:
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
    if signatures[-1].lstrip().rstrip() == "":
        signatures.pop()
    for signature in signatures:
        if text.find(signature.lower()) != -1:
            return text.split(signature.lower(), 1)[0]
    return text


def clean_formatting(text: str) -> str:
    return re.sub(r'^\s+|\n|\r|\s+$', ' ', str(text))


def clean_all_numbers(text: str) -> str:
    return re.sub(r'[0-9]+', '', str(text))


def clean_custom_stop_words(text: str, stop_words: KeywordProcessor) -> str:
    return stop_words.replace_keywords(text.lower()).replace("_EMPTY_", "").strip()


def clean_default_stop_words(text_tokens: list, default_stop_words: set) -> list:
    return [i for i in text_tokens if (i not in default_stop_words)]


def clean_text_with_setting(text: str, setting: PreprocessorSetting, custom_stop_words: KeywordProcessor,
                            default_stop_words: set) -> str:
    return clean_text(text, email_signatures=setting.email_setting.signatures,
                      clean_html=setting.clean_html, clean_email_address=setting.email_setting.clean_address,
                      clean_urls=setting.clean_urls, custom_stop_words=custom_stop_words,
                      default_stop_words=default_stop_words, min_word_len=setting.min_word_len,
                      max_word_len=setting.max_word_len, lemmatize_russian=setting.words_lemmatization_setting.russian,
                      lemmatize_ukrainian=setting.words_lemmatization_setting.ukrainian,
                      min_words_count=setting.min_words_count)


def clean_text(text: str, email_signatures="", clean_html=False, clean_email_address=False, clean_urls=False,
               clean_numbers=True, custom_stop_words=None, default_stop_words=None, min_word_len=0, max_word_len=0,
               lemmatize_russian=False, lemmatize_ukrainian=False,
               min_words_count=0) -> str:
    """
    Execute all cleaner functions

    :return: cleaned text
    """
    if default_stop_words is None:
        default_stop_words = set()
    if email_signatures != "":
        text = clean_email_signature(text, list(email_signatures))
    if clean_html:
        text = clean_html_tags(text)
    if clean_email_address:
        text = clean_email_addresses(text)
    if clean_urls:
        text = clean_url(text)
    # Очищення пробілів та перенесення стрічок
    text = clean_formatting(text)
    if clean_numbers:
        text = clean_all_numbers(text)

    # Проблемні знаки для токенізатора
    # text = text.replace("’", "'").replace('"', "").replace("«", "").replace("»", "").replace("''", "")

    # Очищення стоп слів зі списку stop_words
    if custom_stop_words is not None and len(custom_stop_words) > 0:
        text = clean_custom_stop_words(text, custom_stop_words)

    # Токенізація
    tokens = text_to_word_sequence(text, filters=TOKEN_FILTER)
    # Видалення стандартних стоп слів
    if len(default_stop_words) > 0:
        tokens = clean_default_stop_words(tokens, default_stop_words)
    # Видалення слів < мінімальної кількості символів
    if min_word_len > 0:
        tokens = [i for i in tokens if (len(i) >= int(min_word_len))]
    # Видалення слів > максимальної кількості символів
    if max_word_len > 0:
        tokens = [i for i in tokens if (len(i) <= int(max_word_len))]
    # Лексикологічна нормалізація слів
    if lemmatize_russian or lemmatize_ukrainian:
        tokens = [lemmatize(i, lemmatize_russian, lemmatize_ukrainian) for i in tokens]

    # Очистити по мінімальній кількості слів
    if len(tokens) < int(min_words_count):
        return ""
    text = "".join([i + " " for i in tokens if (len(i) > 0)])
    text = text.lstrip().rstrip()

    return text
