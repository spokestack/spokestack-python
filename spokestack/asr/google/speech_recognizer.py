"""
This module contains the google asr speech recognizer
"""
import logging
from queue import Queue
from threading import Thread
from typing import Any, Generator, Union

import numpy as np
from google.cloud import speech
from google.oauth2 import service_account

from spokestack.context import SpeechContext

_LOG = logging.getLogger(__name__)


class GoogleSpeechRecognizer:
    """Transforms speech into text using Google's ASR.

    Args:
        language (str): The language of given audio as a
                        [BCP-47](https://www.rfc-editor.org/rfc/bcp/bcp47.txt)
                        language tag. Example: "en-US"
        credentials (Union[None, str, dict]): Dictionary of Google API credentials
                                              or path to credentials. if set to None
                                              credentials will be pulled from the
                                              environment variable:
                                              GOOGLE_APPLICATION_CREDENTIALS
        sample_rate (int): sample rate of the input audio (Hz)
        **kwargs (optional): additional keyword arguments
    """

    def __init__(
        self,
        language: str,
        credentials: Union[None, str, dict] = None,
        sample_rate: int = 16000,
        **kwargs: Any,
    ) -> None:
        if credentials:
            if isinstance(credentials, str):
                credentials = service_account.Credentials.from_service_account_file(
                    credentials
                )
            elif isinstance(credentials, dict):
                credentials = service_account.Credentials.from_service_account_info(
                    credentials
                )
            else:
                raise ValueError(
                    "Invalid Credentials: Only dict, str, or None accepted"
                )

        self._client = speech.SpeechClient(credentials=credentials)
        self._config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=language,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )
        self._queue: Queue = Queue()
        self._thread: Any = None

    def __call__(self, context: SpeechContext, frame: np.ndarray) -> None:
        """Main entry point.

        Args:
            context (SpeechContext): current state of the speech pipeline
            frame (np.ndarray): numpy array of PCM-16 audio.

        Returns: None

        """
        if self._thread is None and context.is_active:
            self._begin(context)
        if self._thread is not None and not context.is_active:
            self._commit()
        if context.is_active:
            self._send(frame)

    def _begin(self, context: SpeechContext) -> None:
        self._thread = Thread(
            target=self._receive,
            args=(context,),
        )
        self._thread.start()

    def _receive(self, context: SpeechContext) -> None:
        for response in self._client.streaming_recognize(self._config, self._drain()):
            for result in response.results[:1]:
                for alternative in result.alternatives[:1]:
                    context.transcript = alternative.transcript
                    context.confidence = alternative.confidence
                    if context.transcript:
                        context.event("partial_recognize")

                if result.is_final:
                    if context.transcript:
                        context.event("recognize")
                        _LOG.debug("recognize event")
                    else:
                        context.event("timeout")
                        _LOG.debug("timeout event")

    def _drain(self) -> Generator:
        while True:
            data = self._queue.get()
            if not data:
                break
            yield data

    def _commit(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._thread = None

    def _send(self, frame: np.ndarray) -> None:
        self._queue.put(speech.StreamingRecognizeRequest(audio_content=frame.tobytes()))

    def reset(self) -> None:
        """ resets recognizer """
        if self._thread:
            self._queue.put(None)
            self._thread.join()
            self._thread = None

    def close(self) -> None:
        """ closes recognizer """
        self._client = None
