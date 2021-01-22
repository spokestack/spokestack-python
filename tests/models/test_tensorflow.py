"""
Tests for TFLite model base class
"""
from unittest import mock

import numpy as np

from spokestack.models.tensorflow import TFLiteModel


@mock.patch("spokestack.models.tensorflow.tflite")
def test_inputs(_mock):
    model = TFLiteModel(model_path="model_path")
    model._input_details = [{"name": "inputs", "index": 0}]
    model._output_details = [{"name": "outputs", "index": 3}]

    # test single input
    one = np.zeros((1, 1))
    _ = model(one)

    # test multi input
    one = np.zeros((1, 1))
    _ = model(one, one)

    assert model.output_details
    assert model.input_details


def test_outputs():
    model = mock.MagicMock(return_value=[np.zeros((1, 1))])
    one = np.zeros((1, 1))
    outputs = model(one)
    assert len(outputs) == 1

    # test multi-output
    model = mock.MagicMock(return_value=[np.zeros((1, 1)), np.zeros((1, 1))])
    one = np.zeros((1, 1))
    outputs = model(one)
    assert len(outputs) > 1
