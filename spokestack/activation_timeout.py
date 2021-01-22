"""
This module manages the timeout for speech pipeline activation.
"""
from typing import Any, Union

import numpy as np

from spokestack.context import SpeechContext


class ActivationTimeout:
    """Speech pipeline activation timeout

    Args:
        frame_width (int): frame width of the audio (ms)
        min_active (int): the minimum length of an activation (ms)
        max_active (int): the maximum length of an activation (ms)
    """

    def __init__(
        self,
        frame_width: int = 20,
        min_active: int = 500,
        max_active: int = 5000,
        **kwargs: Any
    ) -> None:

        self._min_active = min_active / frame_width
        self._max_active = max_active / frame_width
        self._is_speech = False
        self._active_length = 0

    def __call__(
        self, context: SpeechContext, frame: Union[np.ndarray, None] = None
    ) -> None:
        """Main entry point that manages timeout

        Args:
            context (SpeechContext): the current state of the pipeline

        """
        vad_fall = self._is_speech and not context.is_speech
        self._is_speech = context.is_speech
        if context.is_active:
            self._active_length += 1
            if self._active_length > self._min_active:
                if vad_fall or self._active_length > self._max_active:
                    self.deactivate(context)

    def deactivate(self, context: SpeechContext) -> None:
        """ Deactivates the speech pipeline """
        self.reset()
        context.is_active = False

    def reset(self) -> None:
        """ Resets active length """
        self.close()

    def close(self) -> None:
        """ Sets active length to zero """
        self._active_length = 0
