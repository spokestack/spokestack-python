"""
This module contains tests for the keyword recognizer.
"""
from unittest import mock

import numpy as np
import pytest

from spokestack.asr.keyword.tflite import KeywordRecognizer
from spokestack.context import SpeechContext


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
            model.input_details = [{"shape": [1, 92, 128]}]
            model.output_details = [{"shape": [1, 3]}]
            model.return_value = [[[0.8, 0.1, 0.3]]]
        return model


@mock.patch("spokestack.asr.keyword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_invalid_args(*args):
    with pytest.raises(ValueError):
        _ = KeywordRecognizer(
            classes=["one", "two", "three"], fft_window_type="hamming"
        )


@mock.patch("spokestack.asr.keyword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_invalid_classes(*args):
    with pytest.raises(ValueError):
        _ = KeywordRecognizer(classes=["one", "two", "three", "four"])

    with pytest.raises(ValueError):
        _ = KeywordRecognizer(classes=["one", "two"])


@mock.patch("spokestack.asr.keyword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_recognize(*args):
    context = SpeechContext()
    recognizer = KeywordRecognizer(classes=["one", "two", "three"])

    test_frame = np.random.rand(
        160,
    ).astype(np.float32)

    context.is_active = True
    for i in range(10):
        recognizer(context, test_frame)
        recognizer(context, test_frame)

    context.is_active = False
    recognizer(context, test_frame)
    assert context.transcript == "one"

    recognizer.close()


@mock.patch("spokestack.asr.keyword.tflite.TFLiteModel", new_callable=ModelFactory)
def test_timeout(*args):
    context = SpeechContext()
    recognizer = KeywordRecognizer(classes=["one", "two", "three"])
    recognizer.detect_model.return_value = [[[0.0, 0.0, 0.0]]]

    test_frame = np.random.rand(
        160,
    ).astype(np.float32)

    context.is_active = True
    for i in range(10):
        recognizer(context, test_frame)
        recognizer(context, test_frame)

    context.is_active = False
    recognizer(context, test_frame)
    assert not context.transcript

    recognizer.close()
