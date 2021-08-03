import re
import json
import logging


def read_config(file):
    with open(file, 'r') as f:
        config = json.load(f)
    # TODO: jsonschema validation
    return config


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


def parse_url(url):
    """Get host and port from url"""
    # Note that this is pretty basic and doesn't check the actual semantics of the host
    #   Don't care about the protocol here
    match = re.match(r'(\w+)://([\w.]+):(\d+)', url)
    protocol, host, port_str = match[1], match[2], match[3]
    return host, int(port_str)
