"""
This module contains tests for the TFLiteNLU class
"""
import json
from unittest import mock

import numpy as np

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
    assert outputs.utterance == utterance
    assert 0.0 <= outputs.confidence <= 1.0
    assert not outputs.slots


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
    model._encode = mock.MagicMock(return_value=[[utterance], [0, 1, 2, 3, 4, 0]])
    outputs = model(utterance)
    assert outputs.utterance == utterance
    assert 0.0 <= outputs.confidence <= 1.0
    assert outputs.slots


@mock.patch("spokestack.nlu.tflite.BertWordPieceTokenizer")
@mock.patch("spokestack.nlu.tflite.TFLiteModel")
def test_classify_with_multiple_slots(_mock_tokenizer, _mock_model, fs):
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
                        },
                        {
                            "name": "number",
                            "capture_name": "number",
                            "description": "",
                            "type": "entity",
                            "facets": "{}",
                        },
                    ],
                }
            ],
            "tags": ["o", "b_test", "i_test", "b_number", "i_number"],
        }
    )
    fs.create_file("/tmp/model/vocab.txt")
    fs.create_file("/tmp/model/metadata.json", contents=metadata)
    model = TFLiteNLU("/tmp/model/")

    model._model._input_details = [{"shape": [1, 5]}]
    model._model._output_details = [{"shape": [1, 1]}, {"shape": [1, 5, 5]}]
    model._model.return_value = [
        np.array([[1.0]]),
        np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
    ]
    model._tokenizer.decode.side_effect = ["test", "ninety nine"]
    utterance = "[CLS] this is test number ninety nine [SEP]"
    model._encode = mock.MagicMock(return_value=[[utterance], [0, 0, 0, 1, 0, 3, 4, 0]])
    outputs = model(utterance)

    assert outputs.utterance == utterance
    assert outputs.intent == "command.test"
    assert 0.0 <= outputs.confidence <= 1.0
    assert outputs.slots == {
        "test": {"name": "test", "parsed_value": "test", "raw_value": "test"},
        "number": {
            "name": "number",
            "parsed_value": "ninety nine",
            "raw_value": "ninety nine",
        },
    }
