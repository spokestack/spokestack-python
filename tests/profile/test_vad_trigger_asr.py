"""
This module contains the tests for vad_trigger_asr profile
"""
from unittest import mock

from spokestack.profile.vad_trigger_asr import VoiceActivityTriggerSpokestackASR


@mock.patch("spokestack.profile.vad_trigger_asr.PyAudioInput")
@mock.patch("spokestack.profile.vad_trigger_asr.SpeechPipeline")
def test_activate(*args):
    pipeline = VoiceActivityTriggerSpokestackASR.create("", "")
    pipeline.start()
    pipeline.run()
