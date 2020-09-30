"""
Mock tests for microphone input
"""
from unittest.mock import patch

import numpy as np  # type: ignore

from spokestack.io.pyaudio import PyAudioInput


@patch("spokestack.io.pyaudio.pyaudio")
def test_audio_input(mock_class):
    mic = PyAudioInput(sample_rate=16000, frame_width=10)
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
