"""
This module contains a context class to manage
state between members of the processing pipeline
"""
import logging
from typing import Callable


_LOG = logging.getLogger(__name__)


class SpeechContext:
    """ Class for managing context of the speech pipeline. """

    def __init__(self) -> None:
        self._is_speech: bool = False
        self._is_active: bool = False
        self._transcript: str = ""
        self._confidence: float = 0.0
        self._handlers: dict = {}

    def add_handler(self, name: str, function: Callable) -> None:
        """Adds a handler to the context

        Args:
            name (str): The name of the event handler
            function (Callable): event handler function

        """
        self._handlers[name] = function

    def event(self, name: str) -> None:
        """Calls the event handler

        Args:
            name (str): The name of the event handler

        """
        handler = self._handlers.get(name)
        if handler:
            handler(self)

    @property
    def is_speech(self) -> bool:
        """This property is to manage if speech is present in the current state
        or not.

        Returns:
            bool: 'True' if is_speech set to 'True', 'False' otherwise
        """
        return self._is_speech

    @is_speech.setter
    def is_speech(self, value: bool) -> None:
        """This method is the setter for the is_speech property.

        Args:
            value (bool): sets is_speech to passed argument
        """
        self._is_speech = value

    @property
    def is_active(self) -> bool:
        """This property manages activity of the context.

        Returns:
            bool: 'True' if context is active, 'False' otherwise.
        """
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """This method sets the is_active property.

        Args:
            value (bool): Boolean to set context activity
        """
        is_active = self._is_active
        self._is_active = value
        if value and not is_active:
            self.event("activate")
            _LOG.info("activate event")
        elif not value and is_active:
            self.event("deactivate")
            _LOG.info("deactivate event")

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
