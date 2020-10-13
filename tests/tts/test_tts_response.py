"""
This module contains tests for tts responses
"""

from unittest import mock

import numpy as np

from spokestack.tts.clients.response import TextToSpeechResponse


def test_response():
    test_url = "https://api.spokestack.io/stream/mp3/test"
    test_content = np.ones(160).tobytes()
    mock_response = mock.Mock(content=test_content, url=test_url)
    mock_response.iter_content.return_value = list(test_content)
    response = TextToSpeechResponse(mock_response)
    assert response.content == test_content
    assert response.iter_content() == list(test_content)
    assert response.url == test_url
