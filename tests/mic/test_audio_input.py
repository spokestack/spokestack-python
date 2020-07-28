"""
Mock tests for microphone input
"""
from unittest.mock import patch

from spokestack.mic.pyaudio import PyAudioMicrophoneInput


@patch("spokestack.mic.pyaudio.pyaudio")
def test_audio_input(mock_class):
    """ Test for microphone input """
    config = {"sample_rate": 16000, "frame_width": 10}
    mic = PyAudioMicrophoneInput(**config)
    mic._audio.open.assert_called()
    # start audio stream
    mic.start()
    assert mic.active
    # read a single frame
    _ = mic()
    mic._stream.read.assert_called()
    # stop audio stream
    mic.stop()
    assert mic.stopped
