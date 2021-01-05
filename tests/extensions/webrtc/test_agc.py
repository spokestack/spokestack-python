"""
This module contains the tests for the AutomaticGainControl extension
"""

import numpy as np  # type: ignore
from spokestack.extensions.webrtc.agc import AutomaticGainControl  # type: ignore

from spokestack.context import SpeechContext


def test_construction():
    _ = AutomaticGainControl(8000, 10)


def test_processing():
    context = SpeechContext()

    sample_rate = 8000
    frequency = 2000

    agc = AutomaticGainControl(
        sample_rate=sample_rate,
        frame_width=10,
        target_level_dbfs=9,
        compression_gain_db=15,
    )

    # valid amplification
    frame = sin_frame(sample_rate, frequency, amplitude=0.08)
    level = rms(frame)
    agc(context, frame)
    assert rms(frame) > level

    # valid attenuation
    frame = sin_frame(sample_rate, frequency)
    level = rms(frame)
    agc(context, frame)
    assert rms(frame) < level

    agc.close()


def sin_frame(sample_rate=16000, frequency=2000, amplitude=1.0):
    frame_width = sample_rate * 10 // 1000
    x = 2 * np.pi * np.arange(sample_rate) / sample_rate
    frame = (np.sin(frequency * x) * amplitude * 32768.0).astype(np.int16)
    return frame[:frame_width]


def rms(y):
    return np.sqrt(np.mean(y.astype(np.float32) ** 2))
