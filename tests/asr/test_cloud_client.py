"""
This module contains the tests for the cloud-based asr client
"""
import json
from unittest import mock

import numpy as np  # type: ignore
import pytest

from spokestack.asr.cloud_client import APIError, CloudClient


@mock.patch("spokestack.asr.cloud_client.WebSocket")
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


@mock.patch("spokestack.asr.cloud_client.WebSocket")
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

    client.close()


@mock.patch("spokestack.asr.cloud_client.WebSocket")
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


@mock.patch("spokestack.asr.cloud_client.WebSocket")
def test_call(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="", idle_timeout=5000)
    client.connect()
    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": True,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )
    client.initialize()
    audio = np.random.rand(160 * 50).astype(np.int16)

    transcript = client(audio)[0]
    assert transcript == "this is a test"


@mock.patch("spokestack.asr.cloud_client.WebSocket")
def test_call_timeout(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="", idle_timeout=5000)
    client.connect()
    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )
    client.initialize()
    audio = np.random.rand(160 * 50).astype(np.int16)

    _ = client(audio)[0]
    assert not client.is_connected


@mock.patch("spokestack.asr.cloud_client.WebSocket")
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


@mock.patch("spokestack.asr.cloud_client.WebSocket")
def test_bad_recv(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="", idle_timeout=5000)
    client.connect()
    client._socket.recv.return_value = json.dumps(
        {
            "error": None,
            "final": False,
            "hypotheses": [{"confidence": 0.5, "transcript": "this is a test"}],
            "status": "ok",
        }
    )
    client.initialize()
    client._socket.recv.return_value = json.dumps({})
    client.receive()
