"""
This module contains the tests for the spokestack text to speech manager
"""
from unittest import mock

import numpy as np  # type: ignore

from spokestack.tts.tts_manager import TextToSpeechManager


def test_synthesize():
    client = mock.MagicMock()
    client.synthesize_speech.return_value = np.zeros(100000, np.int16).tobytes()
    output = mock.MagicMock()

    with mock.patch("spokestack.tts.tts_manager.MP3Decoder"):
        manager = TextToSpeechManager(client, output)
        manager.synthesize(utterance="test utterance")


def test_close():
    client = mock.MagicMock()
    output = mock.MagicMock()
    manager = TextToSpeechManager(client, output)
    manager.close()

    assert not manager._client
    assert not manager._output
