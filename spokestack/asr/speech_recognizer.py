"""
This module contains the recognizer for cloud based ASR
"""

import numpy as np  # type: ignore

from spokestack.asr.cloud_client import CloudClient
from spokestack.context import SpeechContext


class CloudSpeechRecognizer:
    """ speech recognizer """

    def __init__(
        self,
        spokestack_id: str = "",
        spokestack_secret: str = "",
        language: str = "en",
        sample_rate: int = 16000,
        frame_width: int = 10,
    ) -> None:
        self._client_id: str = spokestack_id
        self._secret: str = spokestack_secret
        self._language: str = language
        self._sample_rate: int = sample_rate
        self._frame_width: int = frame_width
        self.max_idle_count = 5000 / frame_width
        self._client: CloudClient = CloudClient(
            self._client_id, self._secret, self._language, sample_rate=self._sample_rate
        )
        self._is_active = False
        self._idle_count: int = 0
        self._final = True
        self._context: SpeechContext

    def __call__(self, context: SpeechContext, frame: np.ndarray) -> None:

        frame = frame.tobytes()
        self._context = context

        if context.is_active and not self._is_active:
            self._begin()
            self._send(frame)
        elif context.is_active:
            self._send(frame)
            self._receive()
        elif self._is_active:
            self._commit()
        elif not self._final:
            self._receive()
        elif self._idle_count < self.max_idle_count:
            self._idle_count += 1
        else:
            self._client.disconnect()

    def _begin(self) -> None:
        if not self._client.is_connected:
            self._client.connect()
        self._client.initialize()
        self._is_active = True
        self._idle_count = 0

    def _send(self, frame) -> None:
        self._client.send(frame)

    def _receive(self):
        self._client.receive()
        self._on_message(self._client.response)

    def _on_message(self, message):
        hypotheses = message.get("hypotheses")
        for hypothesis in hypotheses:
            self._context.transcript = hypothesis.get("transcript")
            self._context.confidence = hypothesis.get("confidence")
        self._final = self._client.response.get("final")

    def _commit(self) -> None:
        self._is_active = False
        self._client.end()

    def reset(self) -> None:
        if self._client.is_connected:
            self._client.disconnect()

        self._idle_count = 0
        self._is_active = False

    def close(self) -> None:
        self._client.close()
