"""
This module contains tests for Voice Activity Detection
"""
from unittest.mock import patch

import numpy as np

from spokestack.context import SpeechContext
from spokestack.vad.webrtc import VoiceActivityDetector


@patch("spokestack.vad.webrtc.webrtcvad")
def test_vad(mock_class):
    context = SpeechContext()
    detector = VoiceActivityDetector(
        sample_rate=16000,
        frame_width=10,
        vad_rise_delay=0,
        vad_fall_delay=20,
        mode="quality",
    )

    # test detection
    frame = np.zeros(160, np.int16).tobytes()
    detector(context, frame)
    assert detector.run_value != 0
    assert detector.run_length >= 1

    detector(context, frame)
    assert detector.run_length >= 2
    assert context.is_speech

    # test reset
    detector.reset()
    assert detector.run_length == 0
    assert detector.run_value == 0
