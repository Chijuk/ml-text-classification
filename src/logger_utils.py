import logging
import os
import sys

FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def init_logging(file_name: str) -> None:
    """ Initialize logging settings

    :param file_name: logfile name
    """
    if file_name[:1] == "\\":
        file_name = file_name[1:]
    if file_name[-4:] != ".log":
        file_name = file_name + ".log"
    path = os.path.abspath(os.fspath(file_name))
    head, tail = os.path.split(path)
    os.makedirs(head, exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(filename=file_name, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FORMATTER)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(FORMATTER)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
