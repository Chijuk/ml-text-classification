import base64
import logging
import re

import bs4 as bs4
from flashtext import KeywordProcessor
from keras.preprocessing.text import text_to_word_sequence

import language_detector as lang
from settings import CleanerSetting, EmailSetting, WordLemmatizationSetting

log = logging.getLogger("text_utils")
TOKEN_FILTER = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


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


def clean_html(text: str) -> str:
    result = ""
    try:
        if text[:2] == "PG":  # Если BASE64поле
            soup = bs4.BeautifulSoup(base64.b64decode(text), "html.parser")
            for cd in soup.findAll(text=True):
                # Берем первый CDATA, т.к. именно в нем HTML текст
                if isinstance(cd, bs4.CData) and len(cd) > 3:
                    cd_string = cd.string
                    result = base64.b64decode(cd_string).decode('utf-16')
                    break
            # Убираем комментарии
            soup = bs4.BeautifulSoup(result, "html.parser")
            for cd in soup.findAll(text=True):
                if isinstance(cd, bs4.Comment):
                    cd.extract()
            for cd in soup.findAll('style'):
                cd.extract()
            for cd in soup.findAll('script'):
                cd.extract()
            for cd in soup.findAll('head'):
                cd.extract()
            for cd in soup.findAll('link'):
                cd.extract()
            # Убираем HTML разметку
            result = soup.get_text()
            soup = bs4.BeautifulSoup(result, "html.parser")
            return soup.get_text()
        else:
            return text
    except Exception as e:
        log.error("Can't clean text. Error: " + str(e), exc_info=True)


def clean_email(text: str, settings: EmailSetting) -> str:
    """

    :param text: text
    :param settings: email settings object
    :return: cleaned text
    """
    signatures = settings.signatures
    if signatures[-1].lstrip().rstrip() == "":
        signatures.pop()
    for signature in signatures:
        if text.find(signature.lower()) != -1:
            return text.split(signature.lower(), 1)[0]
    return text


def clean_text(text: str, settings: CleanerSetting, stop_words: KeywordProcessor) -> str:
    """
    Execute all cleaner functions

    :param stop_words: stop words list, may be None
    :param text: text
    :param settings: cleaner settings object
    :return: cleaned text
    """
    if settings.clean_email:
        text = clean_email(text, settings.email_setting)
    if settings.clean_html:
        text = clean_html(text)
    # Очищення пробілів та перенесення стрічок
    text = re.sub(r'^\s+|\n|\r|\s+$', ' ', str(text))
    # Очищення цифр
    text = re.sub(r'[0-9]+', '', str(text))

    # Проблемні знаки для токенізатора
    # text = text.replace("’", "'").replace('"', "").replace("«", "").replace("»", "").replace("''", "")  # todo?

    # Очищення стоп слів зі списку stop_words
    if settings.clean_stop_words and stop_words is not None and len(stop_words) > 0:
        text = stop_words.replace_keywords(text.lower()).replace("_EMPTY_", "").strip()

    # Токенізація
    tokens = text_to_word_sequence(text, filters=TOKEN_FILTER + '’«»')
    # Видалення слів < мінімальної кількості символів
    if settings.min_word_len > 0:
        tokens = [i for i in tokens if (len(i) >= int(settings.min_word_len))]
    # Видалення слів > максимальної кількості символів
    if settings.max_word_len > 0:
        tokens = [i for i in tokens if (len(i) <= int(settings.max_word_len))]
    # Лексикологічна нормалізація слів
    if settings.use_words_lemmatization:
        tokens = [lemmatize(i, settings.words_lemmatization_setting) for i in tokens]

    # Очистити по мінімальній кількості слів
    if len(tokens) < int(settings.min_words_count):
        return ""
    text = "".join([i + " " for i in tokens if (len(i) > 0)])
    text = text.lstrip().rstrip()

    return text
