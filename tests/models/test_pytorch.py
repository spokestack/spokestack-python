"""
Tests for Pytorch model base class
"""
from unittest import mock

import numpy as np

from spokestack.models.pytorch import PyTorchModel


@mock.patch("spokestack.models.pytorch.torch")
def test_inputs(*args):
    sample = np.random.rand(1, 128, 80).astype(np.float32)
    model = PyTorchModel(model_path="torch_model")

    output = model(sample)

    assert output
