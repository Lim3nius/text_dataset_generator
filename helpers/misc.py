"""
File: misc.py
Author: Tomas Lessy
Email: lessy.mot@gmail.com
Github: https://github.com/lim3nius
Description: This file containes set of functions used for debugging
"""

import traceback
import sys
from typing import List
from logging import getLogger

log = getLogger()


def debug_on_exception(exceptions: List[Exception]):
    '''
    debug_on_exception allows user to examine function only when
    specific set of exceptions are caught, if exceptions aren't
    specified, all exceptions are caught.

    :exceptions: List of exception types on which to drop user
        into python debugger
    '''
    if exceptions is None:
        exceptions = [Exception]

    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                for typ in exceptions:
                    if isinstance(e, typ):
                        breakpoint()
                        res = func(*args, **kwargs)
                        break
                else:
                    raise e
            return res
        return inner
    return wrapper


def exit_on_exception(code: int, *, exceptions: List[Exception] = [Exception]):
    '''
    exit_on_exception terminates program execution and exits with specified
    exit code. If exceptions are specified, than program will be terminated
    only of raise exception is one of given exceptions

    :code: code which will be reported back to OS
    :exceptions: list of exceptions type on which will be program execution
        terminated
    '''
    def wrapper(f):
        def inner(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except Exception as e:
                for ex_type in exceptions:
                    if isinstance(e, ex_type):
                        log.error(f'Caught exception: {e}')
                        traceback.print_exc()
                        sys.exit(code)
                else:
                    raise e
        return inner
    return wrapper
