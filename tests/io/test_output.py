"""
This module contains the test for pyaudio speaker output
"""
from unittest import mock

import numpy as np

from spokestack.io.pyaudio import PyAudioOutput


@mock.patch("spokestack.io.pyaudio.pyaudio")
def test_pyaudio(_mock):
    speaker = PyAudioOutput()
    response = MockResponse()

    with mock.patch("spokestack.io.pyaudio.MP3Decoder") as patched:
        patched.return_value = np.zeros(160).tobytes()
        speaker.write(response)


class MockResponse(mock.Mock):
    def iter_content(self):
        return [np.ones(160, np.int16).tobytes()]
