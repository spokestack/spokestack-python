"""
This module contains tests for SpeechContext
"""
from collections import deque

import numpy as np  # type: ignore

from spokestack.context import SpeechContext


def test_context():
    context = SpeechContext()

    # test buffer
    assert isinstance(context.buffer, deque)

    # test append buffer
    assert len(context.buffer) == 0
    frame = np.zeros(160, np.int16)
    context.append_buffer(frame)
    assert len(context.buffer) > 0

    # test clear buffer
    context.clear_buffer()
    assert len(context.buffer) == 0

    # test is_speech
    assert not context.is_speech
    context.is_speech = True
    assert context.is_speech

    # test is_active
    assert not context.is_active
    context.is_active = True
    assert context.is_active

    # test transcript
    assert not context.transcript
    context.transcript = "this is a test"
    assert context.transcript

    # test confidence
    assert context.confidence == 0.0
    context.confidence = 1.0
    assert context.confidence == 1.0

    # test reset
    context.reset()
    assert not context.is_speech
    assert not context.is_active
    assert not context.transcript
    assert context.confidence == 0.0
    assert len(context.buffer) == 0
