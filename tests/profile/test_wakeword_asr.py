"""
This module contains the tests for the wakeword asr profile.
"""
from unittest import mock

from spokestack.profile.wakeword_asr import WakewordASR


@mock.patch("spokestack.profile.wakeword_asr.WakewordTrigger")
@mock.patch("spokestack.profile.wakeword_asr.PyAudioInput")
def test_activate(*args):
    profile = WakewordASR("", "")
    profile._pipeline = mock.Mock()
    profile.start()
    profile.run()
