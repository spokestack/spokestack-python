"""
Tests for TFLite wakeword models
"""
from unittest import mock

import numpy as np
import pytest

from spokestack.context import SpeechContext
from spokestack.wakeword.tflite import WakewordTrigger


class ModelFactory(mock.MagicMock):
    def __call__(self, model_path):
        model = mock.MagicMock()
        if model_path.endswith("filter.tflite"):
            model.input_details = [{"shape": [1, 257]}]
            model.output_details = [{"shape": [1, 40]}]
            model.return_value = [np.zeros((1, 40))]
        elif model_path.endswith("encode.tflite"):
            model.input_details = [{"shape": [1, 1, 40]}, {"shape": [1, 128]}]
            model.output_details = [{"shape": [1, 128]}, {"shape": [1, 128]}]
            model.return_value = [np.zeros((1, 128)), np.zeros((1, 128))]
        elif model_path.endswith("detect.tflite"):
            model.input_details = [{"shape": [1, 100, 128]}]
            model.output_details = [{"shape": [1, 1]}]
            model.return_value = [np.zeros((1, 1))]
        return model


@mock.patch("spokestack.wakeword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_invalid_args(_mock):
    with pytest.raises(ValueError):
        _ = WakewordTrigger(fft_window_type="hamming")


@mock.patch("spokestack.wakeword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_detect_vad_inactive(_mock):
    context = SpeechContext()

    detector = WakewordTrigger(model_dir="wakeword_model")

    test_frame = np.random.rand(
        160,
    ).astype(np.float32)
    context.is_speech = False
    detector(context, test_frame)
    assert not context.is_active


@mock.patch("spokestack.wakeword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_detect_vad_active(_mock):
    context = SpeechContext()

    detector = WakewordTrigger(model_dir="wakeword_model")

    for _ in range(1):
        test_frame = np.random.rand(
            160,
        ).astype(np.float32)
        context.is_speech = True
        detector(context, test_frame)
        assert not context.is_active

    detector.close()


@mock.patch("spokestack.wakeword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_detect_inactive_vad_deactivate(_mock):
    context = SpeechContext()
    detector = WakewordTrigger(model_dir="wakeword_model")

    for _ in range(3):
        test_frame = np.random.rand(
            160,
        ).astype(np.float32)
        context.is_speech = True
        detector(context, test_frame)
        context.is_speech = False
        assert not context.is_active
    detector(context, test_frame)


@mock.patch("spokestack.wakeword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_detect_activate(_mock):
    context = SpeechContext()
    detector = WakewordTrigger(model_dir="wakeword_model")
    detector.detect_model.return_value[0][:] = 0.6

    test_frame = np.random.rand(
        512,
    ).astype(np.float32)
    context.is_speech = True
    detector(context, test_frame)
    context.is_speech = False
    assert context.is_active


@mock.patch("spokestack.wakeword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_detect_active_min_delay(_mock):
    context = SpeechContext()
    detector = WakewordTrigger(model_dir="wakeword_model")

    test_frame = np.random.rand(
        512,
    ).astype(np.float32)
    context.is_speech = True
    detector(context, test_frame)

    detector.detect_model.return_value[0][:] = 1
    detector(context, test_frame)

    context.is_speech = False
    detector(context, test_frame)
    detector(context, test_frame)
    assert context.is_active


@mock.patch("spokestack.wakeword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_detect_manual_min_delay(_mock):
    context = SpeechContext()
    detector = WakewordTrigger(model_dir="wakeword_model")
    detector.detect_model.return_value[0][:] = 1

    context.is_active = True
    test_frame = np.random.rand(
        512,
    ).astype(np.float32)
    detector(context, test_frame)
    detector(context, test_frame)
    detector(context, test_frame)

    assert context.is_active
