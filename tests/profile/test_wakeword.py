"""
This module contains the tests for vad_trigger_asr profile
"""
from unittest import mock

from spokestack.profile.wakeword import SpokestackWakeword


@mock.patch("spokestack.profile.wakeword.PyAudioInput")
@mock.patch("spokestack.profile.wakeword.WakewordTrigger")
@mock.patch("spokestack.profile.wakeword.SpeechPipeline")
def test_activate(*args):
    pipeline = SpokestackWakeword.create("mock_model_dir")
    pipeline.start()
    pipeline.run()
