"""
File: logger.py
Author: Tomas Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: Logger wrapper
"""

import logging

LOG_FORMAT = '%(asctime)s [%(levelname)s]: %(message)s'
DATE_FMT = '%H:%M:%S'
log = logging.getLogger()
string_to_level = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}


def setup_logger(level: str) -> None:
    """
    setup_logger configures logging format and level,
    which is used through out the program

    :level: lowest level to be logged
    :returns: nothing
    """

    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FMT)
    proper_level = string_to_level.get(level, logging.INFO)
    log.setLevel(proper_level)
