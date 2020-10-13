"""
This module contains the tests for the wakeword asr profile.
"""
from unittest import mock

from spokestack.profile.wakeword_asr import WakewordSpokestackASR


@mock.patch("spokestack.profile.wakeword_asr.PyAudioInput")
@mock.patch("spokestack.profile.wakeword_asr.WakewordTrigger")
@mock.patch("spokestack.profile.wakeword_asr.SpeechPipeline")
def test_activate(*args):
    pipeline = WakewordSpokestackASR.create("", "")
    pipeline.start()
    pipeline.run()
