"""
This module contains the tests for the spokestack text to speech manager
"""
import pytest
from unittest import mock

import numpy as np

from spokestack.tts.manager import SequenceIO, TextToSpeechManager, FORMAT_PCM16


def test_invalid():
    client = mock.MagicMock()
    output = mock.MagicMock()
    with pytest.raises(ValueError):
        TextToSpeechManager(client, output, format_="invalid")


def test_synthesize_mp3():
    client = mock.MagicMock()
    client.synthesize.return_value = np.zeros(100000, np.int16).tobytes()
    output = mock.MagicMock()

    with mock.patch("spokestack.tts.manager.MP3Decoder") as patched:
        patched.return_value = np.zeros(160).tobytes()
        manager = TextToSpeechManager(client, output)
        manager.synthesize(utterance="test utterance")


def test_synthesize_raw():
    audio = np.zeros(16000, np.int16)

    client = mock.MagicMock()
    client.synthesize.return_value = [audio]
    output = mock.MagicMock()

    manager = TextToSpeechManager(client, output, format_=FORMAT_PCM16)
    manager.synthesize(utterance="test utterance")

    output.write.assert_called_with(audio.tobytes())


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
