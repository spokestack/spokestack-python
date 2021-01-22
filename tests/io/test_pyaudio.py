"""
This module contains the tests for pyaudio input/output
"""
from unittest import mock

import numpy as np

from spokestack.io import pyaudio


@mock.patch("spokestack.io.pyaudio.pyaudio")
def test_input(*args):
    mic = pyaudio.PyAudioInput(sample_rate=16000, frame_width=10)
    mic._stream.read.return_value = np.zeros(160, np.int16).tobytes()
    mic._audio.open.assert_called()
    # start audio stream
    mic.start()
    assert mic.is_active
    # read a single frame
    _ = mic.read()
    mic._stream.read.assert_called()
    # stop audio stream
    mic.stop()
    assert mic.is_stopped
    mic.close()


@mock.patch("spokestack.io.pyaudio.pyaudio")
def test_output(*args):
    speaker = pyaudio.PyAudioOutput()
    audio = np.ones(160, np.int16).tobytes()
    speaker.write(audio)
