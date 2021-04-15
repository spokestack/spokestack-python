"""
This module contains the tests for the TTSClient class
"""
from unittest import mock

import numpy as np
import pytest
from requests import Response

from spokestack.tts.clients.spokestack import TextToSpeechClient, TTSError


def test_graphql():
    client = TextToSpeechClient("", "", "")
    voice = "voice"
    profile = "test"

    for mode in ["text", "ssml", "markdown"]:
        method = f"synthesize{mode[0].upper()}{mode[1:]}"
        body = client._build_body("test", mode=mode, voice=voice, profile=profile)
        assert f"{method}(" in body
        assert profile.upper() in body


def test_synthesize_text():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        mock_iterable = mock.MagicMock(
            spec=Response().iter_content(), return_value=test
        )
        patched.post.return_value = MockResponse(status_code=200)
        patched.get.return_value = mock.Mock(
            iter_content=mock_iterable, status_code=200
        )
        response = client.synthesize("test utterance")
        assert response == test
        response = client.synthesize("test utterance", profile="alexa")
        assert response == test


def test_synthesize_ssml():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        mock_iterable = mock.MagicMock(
            spec=Response().iter_content(), return_value=test
        )
        patched.post.return_value = MockResponse(status_code=200)
        patched.get.return_value = mock.Mock(
            iter_content=mock_iterable, status_code=200
        )
        response = client.synthesize("<speak> test utterance </speak>", mode="ssml")
        assert response == test


def test_synthesize_markdown():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        mock_iterable = mock.MagicMock(
            spec=Response().iter_content(), return_value=test
        )
        patched.post.return_value = MockResponse(status_code=200)
        patched.get.return_value = mock.Mock(
            iter_content=mock_iterable, status_code=200
        )
        response = client.synthesize("# test utterance", mode="markdown")
        assert response == test


def test_synthesize_url():
    client = TextToSpeechClient("", "", "")

    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        mock_url = "https://test"
        patched.post.return_value = MockResponse(
            status_code=200,
            return_value={"data": {"synthesizeText": {"url": mock_url}}},
        )
        response = client.synthesize_url("# test utterance", mode="text")
        assert response == mock_url


def test_synthesize_invalid_mode():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.get.return_value = mock.Mock(content=test)
        with pytest.raises(ValueError):
            _ = client.synthesize("test utterance", mode="python")


def test_error_response():
    client = TextToSpeechClient("", "", "")

    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.post.return_value = MockResponse(
            status_code=200,
            return_value={
                "data": {"synthesizeSSML": None},
                "errors": [
                    {
                        "locations": [{"column": 0, "line": 3}],
                        "message": "synthesis_failed",
                        "path": ["synthesizeSSML"],
                    }
                ],
            },
        )

        with pytest.raises(TTSError):
            _ = client.synthesize("utterance")


def test_post_http_error():
    client = TextToSpeechClient("", "", "")

    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.post.return_value = MockResponse(status_code=201)
        with pytest.raises(Exception):
            _ = client.synthesize("utterance")


def test_get_http_error():
    client = TextToSpeechClient("", "", "")

    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.post.return_value = MockResponse(status_code=200)
        patched.get.return_value = MockResponse(status_code=201)
        with pytest.raises(Exception):
            _ = client.synthesize("utterance")


class MockResponse(mock.MagicMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def json(self):
        return self.return_value
