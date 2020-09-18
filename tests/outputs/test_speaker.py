"""
This module contains the test for pyaudio speaker output
"""
from unittest import mock

import numpy as np

from spokestack.outputs.speaker import PyAudioSpeakerOutput


@mock.patch("spokestack.outputs.speaker.pyaudio")
def test_speaker(_mock):
    speaker = PyAudioSpeakerOutput()
    audio = np.ones(160, np.int16).tobytes()
    speaker.write(audio)
