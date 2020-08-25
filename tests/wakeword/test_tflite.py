"""
Tests for TFLite wakeword models
"""
from unittest import mock

import numpy as np  # type: ignore
import pytest  # type: ignore

from spokestack.context import SpeechContext
from spokestack.wakeword.tflite import WakewordDetector


@mock.patch("spokestack.models.tensorflow.tflite")
def test_invalid_args(_mock):
    with pytest.raises(ValueError):
        _ = WakewordDetector(fft_window_type="hamming")


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_vad_inactive(_mock):
    context = SpeechContext()

    detector = WakewordDetector(model_dir="wakeword_model")

    detector.encode_model = mock.MagicMock(
        return_value=[np.ones((1, 1, 128)), np.zeros((1, 128))]
    )
    detector.filter_model = mock.MagicMock(return_value=[np.ones((1, 40))])
    detector.detect_model = mock.MagicMock(return_value=[0.0])
    detector.sample_window = mock.MagicMock()

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
    detector.sample_window = mock.MagicMock()
    detector.encode_window = mock.MagicMock()
    detector.frame_window = mock.MagicMock()

    detector.state = np.zeros((1, 128))

    for _ in range(1):
        test_frame = np.random.rand(160,).astype(np.float32)
        context.is_speech = True
        detector(context, test_frame)
        assert not context.is_active

    detector.reset()


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_inactive_vad_deactivate(_mock):
    context = SpeechContext()
    detector = WakewordDetector(model_dir="wakeword_model")
    detector.encode_model = mock.MagicMock(
        return_value=[np.ones((1, 1, 128)), np.zeros((1, 128))]
    )
    detector.filter_model = mock.MagicMock(return_value=[np.ones((1, 40))])
    detector.detect_model = mock.MagicMock(return_value=[0.0])
    detector.sample_window = mock.MagicMock()
    detector.encode_window = mock.MagicMock()
    detector.frame_window = mock.MagicMock()

    for _ in range(3):
        test_frame = np.random.rand(160,).astype(np.float32)
        context.is_speech = True
        detector(context, test_frame)
        context.is_speech = False
        assert not context.is_active


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_activate(_mock):
    context = SpeechContext()
    detector = WakewordDetector(model_dir="wakeword_model")
    detector.window_size = 512
    detector.encode_model = mock.MagicMock(
        return_value=[np.ones((1, 1, 128)), np.zeros((1, 128))]
    )
    detector.filter_model = mock.MagicMock(return_value=[np.ones((1, 40))])
    detector.detect_model = mock.MagicMock(return_value=[1.0])
    detector.sample_window = mock.MagicMock()
    detector.encode_window = mock.MagicMock()
    detector.frame_window = mock.MagicMock()

    test_frame = np.random.rand(160,).astype(np.float32)
    context.is_speech = True
    detector(context, test_frame)
    context.is_speech = False
    assert context.is_active


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_active_min_delay(_mock):
    context = SpeechContext()
    detector = WakewordDetector(model_dir="wakeword_model")
    detector.encode_model = mock.MagicMock(
        return_value=[np.ones((1, 1, 128)), np.zeros((1, 128))]
    )
    detector.filter_model = mock.MagicMock(return_value=[np.ones((1, 40))])
    detector.detect_model = mock.MagicMock(return_value=[0.0])
    detector.sample_window = mock.MagicMock()
    detector.encode_window = mock.MagicMock()
    detector.frame_window = mock.MagicMock()

    test_frame = np.random.rand(512,).astype(np.float32)
    context.is_speech = True
    detector(context, test_frame)

    detector.detect_model.return_value = [1.0]
    detector(context, test_frame)

    context.is_speech = False
    detector(context, test_frame)
    detector(context, test_frame)
    assert context.is_active


@mock.patch("spokestack.models.tensorflow.tflite")
def test_detect_manual_min_delay(_mock):
    context = SpeechContext()
    detector = WakewordDetector(model_dir="wakeword_model")

    detector.encode_model = mock.MagicMock(
        return_value=[np.ones((1, 1, 128)), np.zeros((1, 128))]
    )
    detector.filter_model = mock.MagicMock(return_value=[np.ones((1, 40))])
    detector.detect_model = mock.MagicMock(return_value=[1.0])
    detector.sample_window = mock.MagicMock()
    detector.encode_window = mock.MagicMock()
    detector.frame_window = mock.MagicMock()

    context.is_active = True
    test_frame = np.random.rand(512,).astype(np.float32)
    detector(context, test_frame)
    detector(context, test_frame)
    detector(context, test_frame)

    assert context.is_active
