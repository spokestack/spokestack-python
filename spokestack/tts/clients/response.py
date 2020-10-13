"""
This module contains text to speech response classes
"""
from typing import Iterator


class TextToSpeechResponse:
    """ Spokestack TTS Client Response Wrapper """

    def __init__(self, response) -> None:
        self._response = response

    def iter_content(self, chunk_size=None) -> Iterator[bytes]:
        """ Iterator containing content of response. """
        return self._response.iter_content(chunk_size)

    @property
    def content(self) -> bytes:
        """ Content of response. """
        return self._response.content

    @property
    def url(self) -> str:
        """ URL to response content. """
        return self._response.url
