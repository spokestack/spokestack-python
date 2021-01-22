"""
This module contains tests for Voice Activity Detection
"""
import numpy as np
import pytest

from spokestack import utils
from spokestack.context import SpeechContext
from spokestack.vad.webrtc import VoiceActivityDetector, VoiceActivityTrigger


def test_invalid_frame_width():
    with pytest.raises(ValueError):
        _ = VoiceActivityDetector(frame_width=30)


def test_invalid_sample_rate():
    with pytest.raises(ValueError):
        _ = VoiceActivityDetector(sample_rate=9000)


def test_invalid_dtype():
    context = SpeechContext()
    detector = VoiceActivityDetector()

    bad_frame = np.random.rand(160)
    with pytest.raises(Exception):
        detector(context, bad_frame)


def test_vad_is_triggered():
    context = SpeechContext()
    detector = VoiceActivityDetector(frame_width=10)

    frame = silence_frame()
    detector(context, frame)
    assert not context.is_speech

    frame = voice_frame()
    detector(context, frame)
    assert context.is_speech

    detector.close()


def test_vad_rise_delay():
    context = SpeechContext()
    detector = VoiceActivityDetector(frame_width=10, vad_rise_delay=30)
    for i in range(3):
        frame = voice_frame()
        detector(context, frame)
        if i < 2:
            assert not context.is_speech
        else:
            assert context.is_speech
    detector.close()


def test_vad_fall_triggered():
    context = SpeechContext()
    detector = VoiceActivityDetector(frame_width=10, vad_fall_delay=20)

    frame = voice_frame()
    detector(context, frame)
    assert context.is_speech

    frame = silence_frame()
    detector(context, frame)
    assert context.is_speech

    frame = voice_frame()
    detector(context, frame)
    assert context.is_speech

    detector.close()


def test_vad_fall_untriggered():
    context = SpeechContext()
    detector = VoiceActivityDetector(frame_width=10, vad_fall_delay=20)

    voice = voice_frame()
    silence = silence_frame()

    detector(context, voice)
    assert context.is_speech

    for i in range(10):
        detector(context, silence)
        assert context.is_speech

    detector(context, silence)
    assert not context.is_speech
    detector.close()


def test_voice_activity_trigger():
    context = SpeechContext()
    trigger = VoiceActivityTrigger()

    frame = voice_frame()

    trigger(context, frame)
    assert not context.is_active

    context.is_speech = True
    trigger(context, frame)
    assert context.is_active

    trigger.close()


def voice_frame(sample_rate=16000, frequency=2000, frame_width=10):
    frame_width = sample_rate * frame_width // 1000
    x = 2 * np.pi * np.arange(sample_rate) / sample_rate
    frame = np.sin(frequency * x)
    frame = frame[:frame_width]
    return utils.float_to_int16(frame)


def silence_frame():
    return np.zeros(160, np.int16)
