"""
This module contains the tests for the GoogleSpeechRecognizer class
"""
from unittest import mock

import numpy as np
import pytest

from spokestack.asr.google.speech_recognizer import GoogleSpeechRecognizer
from spokestack.context import SpeechContext


@mock.patch("spokestack.asr.google.speech_recognizer.speech")
@mock.patch("spokestack.asr.google.speech_recognizer.service_account")
def test_recognize(*args):
    context = SpeechContext()
    audio = np.zeros(160).astype(np.int16)
    recognizer = GoogleSpeechRecognizer(language="en-US", credentials="")

    context.is_active = True
    for i in range(10):
        if i > 3:
            context.is_active = False
        recognizer(context, audio)

    recognizer.reset()
    recognizer.close()


@mock.patch("spokestack.asr.google.speech_recognizer.speech")
@mock.patch("spokestack.asr.google.speech_recognizer.service_account")
def test_receive(*args):
    context = SpeechContext()
    audio = np.zeros(160).astype(np.int16)
    recognizer = GoogleSpeechRecognizer(language="en-US", credentials="")
    recognizer._queue.put([audio, audio, audio])

    recognizer._client.streaming_recognize.return_value = [
        mock.Mock(
            results=[
                mock.Mock(alternatives=[mock.Mock(transcript="test", confidence=0.99)])
            ]
        )
    ]

    context.is_active = True
    for i in range(10):
        if i > 3:

            context.is_active = False
        recognizer(context, audio)

    recognizer._thread = mock.Mock()
    recognizer.reset()
    recognizer.close()


@mock.patch("spokestack.asr.google.speech_recognizer.speech")
@mock.patch("spokestack.asr.google.speech_recognizer.service_account")
def test_drain(*args):
    audio = np.zeros(160).astype(np.int16)
    recognizer = GoogleSpeechRecognizer(language="en-US", credentials="")
    recognizer._queue.put([audio, audio, audio])
    next(recognizer._drain())


@mock.patch("spokestack.asr.google.speech_recognizer.speech")
@mock.patch("spokestack.asr.google.speech_recognizer.service_account")
def test_invalid_creds(*args):
    with pytest.raises(ValueError):
        _ = GoogleSpeechRecognizer(language="en-US", credentials=1234)
