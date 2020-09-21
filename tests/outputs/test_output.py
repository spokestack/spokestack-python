"""
This module contains the test for pyaudio speaker output
"""
from unittest import mock

import numpy as np

from spokestack.outputs.pyaudio import PyAudioOutput


@mock.patch("spokestack.outputs.pyaudio.pyaudio")
def test_pyaudio(_mock):
    speaker = PyAudioOutput()
    audio = np.ones(160, np.int16).tobytes()
    speaker.write(audio)
