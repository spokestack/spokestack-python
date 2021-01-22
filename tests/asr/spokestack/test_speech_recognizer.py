"""
This module tests the cloud speech recognizer
"""
import json
from unittest import mock

import numpy as np

from spokestack.asr.spokestack.speech_recognizer import CloudSpeechRecognizer
from spokestack.context import SpeechContext


def test_recognize():
    context = SpeechContext()
    recognizer = CloudSpeechRecognizer()
    recognizer._client._socket = mock.MagicMock()

    recognizer._client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )

    frame = np.random.rand(160).astype(np.int16)
    # call with context active to test _begin and first _send
    context.is_active = True
    recognizer(context, frame)

    # call again to test with internal _is_active as True
    recognizer(context, frame)

    # call with context not active to test _commit
    context.is_active = False
    recognizer(context, frame)

    recognizer._client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": True,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )

    # call with the client indicating it's the final frame to test _receive
    recognizer(context, frame)

    recognizer._client._socket.max_idle_time = 500
    # test timeout
    for i in range(501):
        recognizer(context, frame)

    assert not context.is_active
    assert not recognizer._client.is_connected


def test_response():
    context = SpeechContext()
    recognizer = CloudSpeechRecognizer()
    recognizer._client._socket = mock.MagicMock()

    recognizer._client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )

    frame = np.random.rand(160).astype(np.int16)

    # run through all the steps
    context.is_active = True
    recognizer(context, frame)
    recognizer(context, frame)
    context.is_active = False
    recognizer(context, frame)

    recognizer._client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": True,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )
    # process the final frame with the final transcript
    recognizer(context, frame)

    assert context.transcript == "this is a test"
    assert context.confidence == 0.5

    recognizer.close()


def test_reset():
    context = SpeechContext()
    recognizer = CloudSpeechRecognizer()
    recognizer._client._socket = mock.MagicMock()

    recognizer._client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )

    frame = np.random.rand(160).astype(np.int16)

    # trigger _begin and first _send
    context.is_active = True
    recognizer(context, frame)

    # trigger _send
    recognizer(context, frame)

    # we haven't triggered _commit or sent the final frame
    # which means context is still active and _is_active is True
    recognizer.reset()

    assert not recognizer._is_active
    assert not recognizer._client.is_connected


def test_empty_transcript():
    context = SpeechContext()
    recognizer = CloudSpeechRecognizer()
    recognizer._client._socket = mock.MagicMock()

    recognizer._client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": ""}],
            "status": "ok",
        }
    )

    frame = np.random.rand(160).astype(np.int16)

    # run through all the steps
    context.is_active = True
    recognizer(context, frame)
    recognizer(context, frame)
    context.is_active = False
    recognizer(context, frame)

    recognizer._client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": True,
            "hypotheses": [{"confidence": 0.5, "transcript": ""}],
            "status": "ok",
        }
    )
    # process the final frame with the final transcript
    recognizer(context, frame)

    assert not context.transcript
    assert context.confidence == 0.5

    recognizer.close()
