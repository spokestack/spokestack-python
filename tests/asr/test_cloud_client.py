"""
This module contains the tests for the cloud-based asr client
"""
import json
from unittest import mock

import numpy as np  # type: ignore
import pytest  # type: ignore

from spokestack.asr.cloud_client import CloudClient


@mock.patch("spokestack.asr.cloud_client.WebSocket")
def test_socket_connect(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")

    assert not client.is_connected

    with pytest.raises(ConnectionError):
        client.initialize()

    client.connect()
    assert client.is_connected

    client.initialize()

    with pytest.raises(ConnectionError):
        client.connect()

    client.disconnect()
    assert not client.is_connected

    with pytest.raises(ConnectionError):
        client.initialize()

    client.disconnect()
    assert not client.is_connected

    client.close()


@mock.patch("spokestack.asr.cloud_client.WebSocket")
def test_send_audio(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    frame = np.random.rand(160).astype(np.int16)

    client.connect()
    client.end()

    for i in range(100):
        client.send(frame)
    client.end()

    client.close()


@mock.patch("spokestack.asr.cloud_client.WebSocket")
def test_reponse_event(_mock):
    client = CloudClient(socket_url="", key_id="", key_secret="")
    client.connect()

    assert not client.response["error"]
    assert not client.response["final"]
    assert not len(client.response["hypotheses"]) > 0
    assert not client.response["status"]

    client.receive()
    assert not client.response["error"]
    assert not client.response["final"]
    assert not len(client.response["hypotheses"]) > 0
    assert not client.response["status"]

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
