"""
This module contains a context class to manage
state between members of the procesing pipeline
"""
from collections import deque
from typing import Deque


class SpeechContext:
    """
    Class for managing context of the speech pipeline
    """

    def __init__(self, **kwargs) -> None:
        self._speech: bool = False
        self._active: bool = False
        self._managed: bool = False
        self._transcript: str = ""
        self._confidence: float = 0.0
        self._buffer: deque = deque()

    @property
    def buffer(self) -> Deque[bytes]:
        return self._buffer

    def append_buffer(self, frame: bytes) -> None:
        self._buffer.append(frame)

    def clear_buffer(self) -> None:
        self._buffer.clear()

    @property
    def is_speech(self) -> bool:
        return self._speech

    @is_speech.setter
    def is_speech(self, value: bool) -> None:
        self._speech = value

    @property
    def is_active(self) -> bool:
        return self._active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._active = value

    @property
    def transcript(self) -> str:
        return self._transcript

    @transcript.setter
    def transcript(self, value: str) -> None:
        self._transcript = value

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self._confidence = value

    def reset(self) -> None:
        self.is_speech = False
        self.is_active = False
        self.transcript = ""
        self.confidence = 0.0
        self.clear_buffer()
