"""
This module contains the Result class for the NLU.
"""
from typing import Any, Dict


class Result:
    """ Convenience wrapper for NLU Results"""

    def __init__(self, **kwargs) -> None:
        self._utterance: str = kwargs.pop("utterance")
        self._intent: str = kwargs.pop("intent")
        self._confidence: float = kwargs.pop("confidence")
        self._slots: Dict[str, Any] = kwargs.pop("slots")

    @property
    def utterance(self) -> str:
        """ Transcript passed to the NLU model """
        return self._utterance

    @property
    def intent(self) -> str:
        """ Classified user intent """
        return self._intent

    @property
    def confidence(self) -> float:
        """ Model confidence in intent classification """
        return self._confidence

    @property
    def slots(self) -> Dict[str, Any]:
        """ Slots found by model """
        return self._slots
