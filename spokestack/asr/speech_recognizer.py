"""
This module contains the recognizer for cloud based ASR in
the speech pipeline
"""

import numpy as np  # type: ignore

from spokestack.asr.cloud_client import CloudClient
from spokestack.context import SpeechContext


class CloudSpeechRecognizer:
    """ Speech recognizer for use in the speech pipeline

    Args:
        spokestack_id (str): identity under spokestack api credentials
        spokestack_secret (str): secret key from spokestack api credentials
        language (str): language recognized
        sample_rate (int): audio sample rate (kHz)
        frame_width (int): frame width of the audio (ms)
        idle_timeout (int): the number of iterations before the connection times out
    """

    def __init__(
        self,
        spokestack_id: str = "",
        spokestack_secret: str = "",
        language: str = "en",
        sample_rate: int = 16000,
        frame_width: int = 10,
        idle_timeout: int = 5000,
    ) -> None:

        self._client: CloudClient = CloudClient(
            spokestack_id,
            spokestack_secret,
            language,
            sample_rate=sample_rate,
            idle_timeout=int(idle_timeout / frame_width),
        )
        self._is_active = False

    def __call__(self, context: SpeechContext, frame: np.ndarray) -> None:
        """ Entry point of the recognizer

        Args:
            context (SpeechContext): current state of the speech pipeline
            frame (np.ndarray): single frame of audio

        """

        if context.is_active and not self._is_active:
            self._begin()
            self._send(frame)
        elif context.is_active:
            self._send(frame)
            self._receive(context)
        elif self._is_active:
            self._commit()
        elif not self._is_final:
            self._receive(context)
        elif self._client.idle_count < self._client.idle_timeout:
            self._client.idle_count += 1
        else:
            self._client.disconnect()

    def _begin(self) -> None:
        self._client.connect()
        self._client.initialize()
        self._is_active = True
        self._client.idle_count = 0

    def _send(self, frame) -> None:
        self._client.send(frame)

    def _receive(self, context):
        self._client.receive()
        hypotheses = self._client.response.get("hypotheses")
        if hypotheses:
            hypothesis = hypotheses[0]
            context.transcript = hypothesis["transcript"]
            context.confidence = hypothesis["confidence"]
        self._is_final = self._client.response.get("final")

    def _commit(self) -> None:
        self._is_active = False
        self._client.end()

    def reset(self) -> None:
        """ resets client connection """
        self._client.idle_count = 0
        self._is_active = False
        self.close()

    def close(self) -> None:
        """ closes client connection """
        self._client.disconnect()
