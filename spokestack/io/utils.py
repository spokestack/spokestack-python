"""
This module contains utilities for input/output
"""


class SequenceIO:
    """ Wrapper that allows for incrementally received audio to be decoded. """

    def __init__(self, sequence):
        self._sequence = iter(sequence)

    def read(self, size=None):
        try:
            return next(self._sequence)
        except StopIteration:
            return b""
