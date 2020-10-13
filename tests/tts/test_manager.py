"""
This module contains the tests for the spokestack text to speech manager
"""
from unittest import mock

import numpy as np  # type: ignore

from spokestack.tts.manager import TextToSpeechManager


def test_synthesize():
    client = mock.MagicMock()
    client.synthesize.return_value = np.zeros(100000, np.int16).tobytes()
    output = mock.MagicMock()

    manager = TextToSpeechManager(client, output)
    manager.synthesize(utterance="test utterance", mode="text", voice="demo-male")


def test_close():
    client = mock.MagicMock()
    output = mock.MagicMock()
    manager = TextToSpeechManager(client, output)
    manager.close()

    assert not manager._client
    assert not manager._output
