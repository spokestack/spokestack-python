"""
Tests for TFLite wakeword models
"""
from unittest import mock

import numpy as np  # type: ignore

from spokestack.context import SpeechContext
from spokestack.wakeword.tflite import WakewordDetector


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_vad_inactive(_mock):
    context = SpeechContext()
    detector = WakewordDetector(model_dir="wakeword_model")

    test_frame = np.random.rand(160,).astype(np.float32)
    context.is_speech = False
    detector(context, test_frame)
    assert not context.is_active


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_vad_active(_mock):
    context = SpeechContext()
    detector = WakewordDetector(model_dir="wakeword_model")
    detector.encode_model = mock.MagicMock(
        return_value=[np.ones((1, 1, 128)), np.zeros((1, 128))]
    )
    detector.filter_model = mock.MagicMock(return_value=[np.ones((1, 40))])
    detector.detect_model = mock.MagicMock(return_value=[0.0])

    for _ in range(3):
        test_frame = np.random.rand(480,).astype(np.float32)
        context.is_speech = True
        detector(context, test_frame)
        assert not context.is_active

    detector.reset()
    detector.close()


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_active(_mock):
    context = SpeechContext()
    detector = WakewordDetector(model_dir="wakeword_model")
    detector.encode_model = mock.MagicMock(
        return_value=[np.ones((1, 1, 128)), np.zeros((1, 128))]
    )
    detector.filter_model = mock.MagicMock(return_value=[np.ones((1, 40))])
    detector.detect_model = mock.MagicMock(return_value=[0.6])

    for _ in range(5):
        test_frame = np.random.rand(160,).astype(np.float32)
        context.is_speech = True
        detector(context, test_frame)
        assert context.is_speech
    context.reset()
