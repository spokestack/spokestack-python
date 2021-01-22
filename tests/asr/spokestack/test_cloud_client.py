"""
This module contains the tests for the cloud-based asr client
"""
import json
from unittest import mock

import numpy as np
import pytest

from spokestack.asr.spokestack.cloud_client import APIError, CloudClient


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_socket_connect(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")

    with pytest.raises(ConnectionError):
        client.initialize()

    assert not client.is_connected
    client.connect()

    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )
    assert client.is_connected

    # try a double connect which should be silently ignored
    client.connect()

    client.disconnect()
    assert not client.is_connected


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_send_audio(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    frame = np.random.rand(160).astype(np.int16)

    with pytest.raises(ConnectionError):
        client.send(frame)

    with pytest.raises(ConnectionError):
        client.end()

    client.connect()
    client.end()

    for i in range(100):
        client.send(frame)
    client.end()


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_reponse_event(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")

    with pytest.raises(ConnectionError):
        client.receive()

    client.connect()
    client._socket.recv.return_value = json.dumps(
        {"error": None, "final": False, "hypotheses": [], "status": "ok"}
    )
    client.initialize()
    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )

    client.receive()
    assert not client.response["error"]
    assert not client.response["final"]
    assert len(client.response["hypotheses"]) > 0
    assert client.response["status"] == "ok"

    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )

    client.receive()
    assert not client.response["error"]
    assert not client.response["final"]
    assert client.response["hypotheses"][0]["confidence"] == 0.5
    assert client.response["hypotheses"][0]["transcript"] == "this is a test"
    assert client.response["status"] == "ok"

    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": True,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )

    client.receive()
    assert client.is_final


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_call(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="", idle_timeout=5000)
    client.connect()
    client._socket.recv.side_effect = [
        json.dumps(
            {
                "error": None,
                "final": False,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
        json.dumps(
            {
                "error": None,
                "final": False,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
        json.dumps(
            {
                "error": None,
                "final": False,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
        json.dumps(
            {
                "error": None,
                "final": True,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
    ]
    client.initialize()
    audio = np.random.rand(160 * 50).astype(np.int16)

    transcript = client(audio)[0]
    assert transcript == {"confidence": 0.5, "transcript": "this is a test"}


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_api_error(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="", idle_timeout=5000)

    client.connect()
    client._socket.recv.return_value = json.dumps(
        {
            "error": "invalid_language",
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "error",
        }
    )
    with pytest.raises(APIError):
        client.initialize()


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_bad_recv(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="", idle_timeout=5000)
    client.connect()
    client._socket.recv.side_effect = [
        json.dumps(
            {
                "error": None,
                "final": False,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
        # bad response
        {},
    ]
    client.initialize()
    client.receive()


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_type_conversions(_mock):
    dummpy_inputs = [
        json.dumps(
            {
                "error": None,
                "final": False,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
        json.dumps(
            {
                "error": None,
                "final": False,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
        json.dumps(
            {
                "error": None,
                "final": True,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
    ]
    client = CloudClient(socket_url="", key_id="", key_secret="")
    client.connect()
    client._socket.recv.side_effect = dummpy_inputs

    client.initialize()
    audio = np.zeros((100), np.float32)

    # as np.float32
    transcript = client(audio)[0]
    assert transcript == {"confidence": 0.5, "transcript": "this is a test"}

    # as bytes
    client.connect()
    client._socket.recv.side_effect = dummpy_inputs
    client.initialize()
    transcript = client(audio.tobytes())[0]
    assert transcript == {"confidence": 0.5, "transcript": "this is a test"}

    # as int16
    client.connect()
    client._socket.recv.side_effect = dummpy_inputs
    client.initialize()
    transcript = client(audio.astype(np.int16))[0]
    assert transcript == {"confidence": 0.5, "transcript": "this is a test"}

    # as float64
    client.connect()
    client._socket.recv.side_effect = dummpy_inputs
    client.initialize()
    transcript = client(audio.astype(np.float64))[0]
    assert transcript == {"confidence": 0.5, "transcript": "this is a test"}

    # as invalid
    client.connect()
    client._socket.recv.side_effect = dummpy_inputs
    client.initialize()
    with pytest.raises(TypeError):
        _ = client(audio.astype(np.complex64))[0]


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_is_connected(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    client.connect()
    assert client.is_connected
    client.disconnect()
    assert not client.is_connected


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_initialize(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    client.connect()
    assert client.is_connected
    client._socket.recv.return_value = json.dumps(
        {"error": None, "final": False, "hypotheses": [], "status": "ok"}
    )
    client.initialize()


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_disconnect(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")

    assert not client.is_connected
    client.connect()
    assert client.is_connected
    client.disconnect()
    assert not client.is_connected


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_end(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    client.connect()
    client._socket.recv.side_effect = [
        json.dumps(
            {
                "error": None,
                "final": False,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
        json.dumps(
            {
                "error": None,
                "final": True,
                "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
                "status": "ok",
            }
        ),
    ]

    client.initialize()
    client.end()
    client.receive()
    assert client.is_final


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_response(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    response = {
        "error": None,
        "final": False,
        "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
        "status": "ok",
    }
    client.connect()
    client._socket.recv.return_value = json.dumps(response)
    client.receive()
    assert client.response == response


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_is_final(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    client.connect()
    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": True,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )
    client.receive()
    assert client.is_final


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_idle_timeout(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="", idle_timeout=500)
    assert client.idle_timeout == 500


@mock.patch("spokestack.asr.spokestack.cloud_client.WebSocket")
def test_idle_count(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    assert client.idle_count == 0

    for i in range(5):
        client.idle_count += 1

    assert client.idle_count == 5
