"""
This module contains a context class to manage
state between members of the procesing pipeline
"""
from collections import deque
from typing import Deque


class SpeechContext:
    """Class for managing context of the speech pipeline.

    Args:
        **kwargs
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
        """This property holds the audio.

        Returns:
            Deque[bytes]: a deque of audio frames
        """
        return self._buffer

    def append_buffer(self, frame: bytes) -> None:
        """This method adds audio to the context buffer.

        Args:
            frame (bytes): a frame of audio
        """
        self._buffer.append(frame)

    def clear_buffer(self) -> None:
        """ Empties the current buffer. """
        self._buffer.clear()

    @property
    def is_speech(self) -> bool:
        """This property is to manage if speech is present in the current state
        or not.

        Returns:
            bool: 'True' if is_speech set to 'True', 'False' otherwise
        """
        return self._speech

    @is_speech.setter
    def is_speech(self, value: bool) -> None:
        """This method is the setter for the is_speech property.

        Args:
            value (bool): sets is_speech to passed argument
        """
        self._speech = value

    @property
    def is_active(self) -> bool:
        """This property manages activity of the context.

        Returns:
            bool: 'True' if context is active, 'False' otherwise.
        """
        return self._active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """This method sets the is_active property.

        Args:
            value (bool): Boolean to set context activity
        """
        self._active = value

    @property
    def transcript(self) -> str:
        """This property is the text representation of the audio buffer

        Returns:
            str: the value of the transcript property
        """
        return self._transcript

    @transcript.setter
    def transcript(self, value: str) -> None:
        """This method sets the transcript from a given string.

        Args:
            value (str): The text representation of speech input
        """
        self._transcript = value

    @property
    def confidence(self) -> float:
        """This property contains the confidence of a classification result.

        Returns:
            float: model confidence of classification
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        """This method sets the confidence property.

        Args:
            value (float): model confidence in classification result
        """
        self._confidence = value

    def reset(self) -> None:
        """Resets the context state"""
        self.is_speech = False
        self.is_active = False
        self.transcript = ""
        self.confidence = 0.0
        self.clear_buffer()
