from __future__ import absolute_import, division, print_function, unicode_literals

import sys


class KaffeError(Exception):
    pass


def print_stderr(msg):
    sys.stderr.write('{}\n'.format(msg))
