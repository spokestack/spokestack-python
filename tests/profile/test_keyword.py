"""
This module contains the tests for the keyword profile.
"""
from unittest import mock

from spokestack.profile.keyword import SpokestackKeyword


@mock.patch("spokestack.profile.keyword.PyAudioInput")
@mock.patch("spokestack.profile.keyword.KeywordRecognizer")
@mock.patch("spokestack.profile.keyword.SpeechPipeline")
def test_activate(*args):
    pipeline = SpokestackKeyword.create(
        classes=[
            "one",
            "two",
            "three",
        ],
        model_dir="mock_model_dir",
    )
    pipeline.run()
