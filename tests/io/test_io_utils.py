"""
This module contains tests for input/output utils.
"""
import numpy as np

from spokestack.io.utils import SequenceIO


def test_sequence_io():
    test = (np.ones(1000, np.int16).tobytes() for i in range(10))
    stream = SequenceIO(test)
    for _ in test:
        stream.read()
    # read after StopIteration
    stream.read()
