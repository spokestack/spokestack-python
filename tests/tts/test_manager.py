"""
This module contains the tests for the spokestack text to speech manager
"""
from unittest import mock

import numpy as np

from spokestack.tts.manager import SequenceIO, TextToSpeechManager


def test_synthesize():
    client = mock.MagicMock()
    client.synthesize.return_value = np.zeros(100000, np.int16).tobytes()
    output = mock.MagicMock()

    with mock.patch("spokestack.tts.manager.MP3Decoder") as patched:
        patched.return_value = np.zeros(160).tobytes()
        manager = TextToSpeechManager(client, output)
        manager.synthesize(utterance="test utterance")


def test_close():
    client = mock.MagicMock()
    output = mock.MagicMock()
    manager = TextToSpeechManager(client, output)
    manager.close()

    assert not manager._client
    assert not manager._output


def test_sequence_io():
    test = (np.ones(1000, np.int16).tobytes() for i in range(10))
    stream = SequenceIO(test)
    for _ in test:
        stream.read()
    # read after StopIteration
    stream.read()
