"""
This module contains the tests for vad_trigger_asr profile
"""
from unittest import mock

from spokestack.profile.vad_trigger_asr import VoiceActivityTriggerASR


@mock.patch("spokestack.profile.vad_trigger_asr.PyAudioInput")
def test_activate(*args):
    profile = VoiceActivityTriggerASR("", "")
    profile._pipeline = mock.Mock()
    profile.start()
    profile.run()
