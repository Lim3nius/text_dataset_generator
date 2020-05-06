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

log_level_table = {
    'debug': log.debug,
    'info': log.info,
    'warn': log.warn,
    'error': log.error,
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


def log_function_call(log_level):
    def wrapper(f):
        def inner(*args, **kwargs):
            log_level_table[log_level](
                f'Call to {f.__name__}({args},{kwargs}')
            res = f(*args, **kwargs)
            log_level_table[log_level](f'Return from {f.__name__} -> {res}')
            return res
        return inner
    return wrapper
