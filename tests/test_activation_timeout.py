"""
This module contains the tests for the activation timeout class
"""
from spokestack.activation_timeout import ActivationTimeout
from spokestack.context import SpeechContext


def test_timeout():
    max_active = 500
    context = SpeechContext()
    timeout = ActivationTimeout(max_active=max_active)
    context.is_active = True

    for i in range(max_active + 1):
        timeout(context)

    assert not context.is_active

    timeout.close()
