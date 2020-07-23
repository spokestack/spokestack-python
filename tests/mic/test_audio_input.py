"""
Mock tests for microphone input
"""
from unittest import mock

from spokestack.mic.pyaudio import PyAudioMicrophoneInput
from spokestack.config import SpeechConfig


def test_audio_input():

    params = {"sample_rate": 16000, "frame_width": 10}
    config = SpeechConfig(params)
    mic = PyAudioMicrophoneInput(config)
    mic.audio = mock.MagicMock()
    mic.build()
    # start audio stream
    mic.start()
    assert mic.is_active()
    # read a single frame
    frame = mic.read()
    assert frame
    # stop audio stream
    mic.stop()
    assert mic.is_stopped()
