"""
This module contains the tests for the automatic noise suppression extension.
"""
import numpy as np
import pytest

from spokestack import utils
from spokestack.context import SpeechContext
from spokestack.nsx.webrtc import AutomaticNoiseSuppression

np.random.seed(42)


def test_construction():
    nsx = AutomaticNoiseSuppression(16000, 1)
    nsx.reset()
    nsx.close()


def test_invalid_sample_rate():
    with pytest.raises(ValueError):
        _ = AutomaticNoiseSuppression(sample_rate=900000, policy=1)


def test_invalid_frame_size():
    context = SpeechContext()
    nsx = AutomaticNoiseSuppression(16000, 1)

    bad_frame = np.random.rand(550)
    with pytest.raises(ValueError):
        nsx(context, bad_frame)


def test_invalid_frame_dtype():
    context = SpeechContext()
    nsx = AutomaticNoiseSuppression(16000, 1)

    bad_frame = np.random.rand(320)
    with pytest.raises(TypeError):
        nsx(context, bad_frame)


def test_processing():
    context = SpeechContext()
    nsx = AutomaticNoiseSuppression(16000, 1)

    # no suppression
    expect = sin_frame()
    actual = sin_frame()
    np.allclose(rms(expect), rms(actual), atol=3)

    nsx(context, utils.float_to_int16(sin_frame()))
    nsx(context, utils.float_to_int16(actual))
    np.allclose(rms(expect), rms(actual), atol=3)

    nsx.close()

    # valid suppression
    expect = sin_frame()
    actual = add_noise(sin_frame())
    nsx(context, utils.float_to_int16(sin_frame()))
    nsx(context, utils.float_to_int16(actual))
    np.allclose(rms(expect), rms(actual), atol=3)


def sin_frame(sample_rate=16000, frequency=100, frame_width=20):
    frame_width = sample_rate * frame_width // 1000
    x = 2 * np.pi * np.arange(sample_rate) / sample_rate
    frame = np.sin(frequency * x)
    frame = frame[:frame_width]
    return frame


def rms(y):
    return 20 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-5) / 2e-5)


def add_noise(frame):
    noise = np.random.normal(size=frame.shape)
    noise = noise / np.power(10, 10 / 20.0)
    frame += noise
    return np.clip(frame, -1.0, 1.0)
