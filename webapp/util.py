import logging


def get_verbosity(val):
    """Convert verbosity value into logging level"""
    if val >= 3:
        return logging.DEBUG
    elif val >= 2:
        return logging.INFO
    elif val >= 1:
        return logging.WARNING
    elif val >= 0:
        return logging.ERROR
    else:
        return logging.CRITICAL
