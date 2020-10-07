"""
This module contains the Result class for the NLU.
"""
from typing import Any, Dict

from spokestack.typing import Intent, Slot, Utterance


class Result:
    """ Convenience wrapper for NLU Results"""

    def __init__(self):
        self._utterance: Utterance = None
        self._intent: Intent = None
        self._confidence: float = 0.0
        self._slots: Dict[str, Slot] = {}

    @property
    def utterance(self) -> Utterance:
        """ Transcript passed to the NLU model """
        return self._utterance

    @utterance.setter
    def utterance(self, value: Utterance) -> None:
        """ Sets utterance to given value """
        self._utterance = value

    @property
    def intent(self) -> Intent:
        """ Classified user intent """
        return self._intent

    @intent.setter
    def intent(self, value: Intent) -> None:
        """ Sets intent to given value """
        self._intent = value

    @property
    def confidence(self) -> float:
        """ Model confidence in intent classification """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        """ Sets confidence to given value """
        self._confidence = value

    @property
    def slots(self) -> Dict[str, Any]:
        """ Slots found by model """
        return self._slots

    @slots.setter
    def slots(self, value: Dict[str, Slot]) -> None:
        """ Sets slots to given value """
        self._slots = value
