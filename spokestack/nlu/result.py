"""
This module contains the Result class for the NLU.
"""
from typing import Any, Dict


class Result:
    """Convenience wrapper for NLU Results

    Args:
        utterance (str): original input string
        intent (str): detected user intention
        confidence (float): model confidence in intent prediction
        slots (Dict[str, Any]): specific tokens needed by intent and
                                type information necessary for parsing

    """

    def __init__(
        self, utterance: str, intent: str, confidence: float, slots: Dict[str, Any]
    ) -> None:
        self._utterance: str = utterance
        self._intent: str = intent
        self._confidence: float = confidence
        self._slots: Dict[str, Any] = slots

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
