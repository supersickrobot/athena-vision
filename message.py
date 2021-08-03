import collections

Action = collections.namedtuple('Action', ['name', 'values', 'request_id'], defaults=[None, None, None])
Display = collections.namedtuple('Display', ['name', 'values', 'request_id'], defaults=[None, None, None])
