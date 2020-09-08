"""
This module contains tests for the TFLiteNLU class
"""
import json
from unittest import mock

import numpy as np  # type: ignore

from spokestack.nlu.tflite import TFLiteNLU


@mock.patch("spokestack.nlu.tflite.BertWordPieceTokenizer")
@mock.patch("spokestack.nlu.tflite.TFLiteModel")
def test_classify_without_slots(_mock_tokenizer, _mock_model, fs):
    metadata = json.dumps(
        {
            "domain": "dummy",
            "intents": [
                {
                    "name": "command.test",
                    "description": "",
                    "implicit_slots": [],
                    "slots": [],
                }
            ],
            "tags": ["o", "b_test", "i_test"],
        }
    )
    fs.create_file("/tmp/model/vocab.txt")
    fs.create_file("/tmp/model/metadata.json", contents=metadata)
    model = TFLiteNLU("/tmp/model/")

    model._model._input_details = [{"shape": [1, 50]}]
    model._model._output_details = [{"shape": [1, 1]}, {"shape": [1, 50, 3]}]
    model._model.return_value = [np.random.random((1, 1)), np.random.random((1, 50, 3))]

    utterance = "this is only a test"
    outputs = model(utterance)
    assert outputs["utterance"] == utterance
    assert 0.0 <= outputs["confidence"] <= 1.0
    assert not outputs["slots"]


@mock.patch("spokestack.nlu.tflite.BertWordPieceTokenizer")
@mock.patch("spokestack.nlu.tflite.TFLiteModel")
def test_classify_with_slots(_mock_tokenizer, _mock_model, fs):
    metadata = json.dumps(
        {
            "domain": "dummy",
            "intents": [
                {
                    "name": "command.test",
                    "description": "",
                    "implicit_slots": [],
                    "slots": [
                        {
                            "name": "test",
                            "capture_name": "test",
                            "description": "",
                            "type": "entity",
                            "facets": "{}",
                        }
                    ],
                }
            ],
            "tags": ["o", "b_test", "i_test"],
        }
    )
    fs.create_file("/tmp/model/vocab.txt")
    fs.create_file("/tmp/model/metadata.json", contents=metadata)
    model = TFLiteNLU("/tmp/model/")

    model._model._input_details = [{"shape": [1, 50]}]
    model._model._output_details = [{"shape": [1, 1]}, {"shape": [1, 50, 3]}]
    model._model.return_value = [np.random.random((1, 1)), np.random.random((1, 50, 3))]

    utterance = "this is only a test"
    outputs = model(utterance)
    assert outputs["utterance"] == utterance
    assert 0.0 <= outputs["confidence"] <= 1.0
    assert outputs["slots"]
