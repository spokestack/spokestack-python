"""
This module contains the tests for the activation timeout class
"""
from spokestack.activation_timeout import ActivationTimeout
from spokestack.context import SpeechContext


def test_timeout_vad_fall():
    max_active = 500
    min_active = 20
    context = SpeechContext()
    timeout = ActivationTimeout(min_active=min_active, max_active=max_active)

    context.is_active = True
    context.is_speech = False

    timeout(context)
    context.is_speech = True

    timeout(context)
    assert context.is_active

    context.is_speech = False

    steps_before_timeout = (min_active // 20) + 2
    for _ in range(steps_before_timeout):
        timeout(context)
    assert not context.is_active

    timeout.close()


def test_max_active():
    max_active = 500
    min_active = 20
    context = SpeechContext()
    timeout = ActivationTimeout(min_active=min_active, max_active=max_active)

    context.is_active = True

    steps_before_timeout = (max_active // 20) + 1
    for _ in range(steps_before_timeout):
        timeout(context)

    assert not context.is_active

    timeout.close()


def test_min_active():
    max_active = 500
    min_active = 120
    context = SpeechContext()
    timeout = ActivationTimeout(min_active=min_active, max_active=max_active)

    context.is_active = True

    # call with speech active
    context.is_speech = True
    timeout(context)

    # call timeout after speech is no longer detected
    context.is_speech = False
    timeout(context)
    assert context.is_active

    # vad fall should be True
    # with context still active
    timeout(context)
    assert context.is_active

    # context should remain active until min active
    steps_before_deactivate = min_active // 20
    for _ in range(steps_before_deactivate):
        timeout(context)
        assert context.is_active

    # call with speech active
    context.is_speech = True
    timeout(context)

    # call timeout after speech is no longer detected
    # min active should be satisfied
    context.is_speech = False
    timeout(context)
    assert not context.is_active

    timeout.close()
