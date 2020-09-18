"""
This module contains the tests for the TTSClient class
"""
from unittest import mock

import numpy as np  # type: ignore
import pytest

from spokestack.tts.clients.spokestack import TTSError, TextToSpeechClient


def test_synthesize_text():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.get.return_value = mock.Mock(content=test)
        response = client.synthesize_speech("test utterance")
        assert response == test


def test_synthesize_ssml():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.get.return_value = mock.Mock(content=test)
        response = client.synthesize_speech(
            "<speak> test utterance </speak>", mode="ssml"
        )
        assert response == test


def test_synthesize_markdown():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.get.return_value = mock.Mock(content=test)
        response = client.synthesize_speech("# test utterance", mode="markdown")
        assert response == test


def test_synthesize_invalid_mode():
    client = TextToSpeechClient("", "", "")

    test = np.ones(160).tobytes()
    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.get.return_value = mock.Mock(content=test)
        with pytest.raises(ValueError):
            response = client.synthesize_speech("test utterance", mode="python")
            assert response == test


def test_error_response():
    client = TextToSpeechClient("", "", "")

    with mock.patch("spokestack.tts.clients.spokestack.requests") as patched:
        patched.post.return_value = MockResponse(
            return_value={
                "data": {"synthesizeSSML": None},
                "errors": [
                    {
                        "locations": [{"column": 0, "line": 3}],
                        "message": "synthesis_failed",
                        "path": ["synthesizeSSML"],
                    }
                ],
            }
        )

        with pytest.raises(TTSError):
            _ = client.synthesize_speech("utterance")


class MockResponse(mock.MagicMock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def json(self):
        return self.return_value
