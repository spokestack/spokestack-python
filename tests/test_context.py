"""
This module contains tests for SpeechContext
"""

from spokestack.context import SpeechContext


def test_context():
    context = SpeechContext()

    # test is_speech
    assert not context.is_speech
    context.is_speech = True
    assert context.is_speech

    # test is_active
    assert not context.is_active
    context.is_active = True
    assert context.is_active

    # test transcript
    assert not context.transcript
    context.transcript = "this is a test"
    assert context.transcript

    # test confidence
    assert context.confidence == 0.0
    context.confidence = 1.0
    assert context.confidence == 1.0

    # test reset
    context.reset()
    assert not context.is_speech
    assert not context.is_active
    assert not context.transcript
    assert context.confidence == 0.0


def test_handler():
    def on_speech(context):
        context.transcript = "event handled"

    context = SpeechContext()
    context.add_handler("recognize", on_speech)

    context.event("recognize")

    assert context.transcript == "event handled"
