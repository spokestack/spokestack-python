"""
This module contains the webrtc component for voice activity detection (vad)
"""
import logging
from typing import Any

import numpy as np

from spokestack.context import SpeechContext
from spokestack.extensions.webrtc.vad import WebRtcVad

QUALITY = 0
LOW_BITRATE = 1
AGGRESSIVE = 2
VERY_AGGRESSIVE = 3

_LOG = logging.getLogger(__name__)


class VoiceActivityDetector:
    """This class detects the presence of voice in a frame of audio.

    Args:
        sample_rate (int): sample rate of the audio (Hz)
        frame_width (int): width of the audio frame: 10, 20, or 30 (ms)
        vad_rise_delay (int): rising edge delay (ms)
        vad_fall_delay (int): falling edge delay (ms)
        mode (int): named constant to set mode for vad

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_width: int = 20,
        vad_rise_delay: int = 0,
        vad_fall_delay: int = 0,
        mode: int = QUALITY,
        **kwargs: Any
    ) -> None:

        # validate sample rate
        self._sample_rate: int = sample_rate
        if self._sample_rate not in {8000, 16000, 32000}:
            raise ValueError("invalid_sample_rate")
        self._frame_width: int = frame_width
        # validate frame width
        if self._frame_width not in {10, 20}:
            raise ValueError("invalid_frame_width")

        self._rise_length: int = vad_rise_delay // frame_width
        self._fall_length: int = vad_fall_delay // frame_width

        self._vad = WebRtcVad(sample_rate=sample_rate, mode=mode)

        self._run_value: int = 0
        self._run_length: int = 0

    def __call__(self, context: SpeechContext, frame: np.ndarray) -> None:
        """Processes a single frame of audio to determine if voice is present

        Args:
            context (SpeechContext): State based information that needs to be shared
            between pieces of the pipeline
            frame (np.ndarray): Single frame of PCM-16 audio from an input source

        """
        # validate dtype
        if not np.issubdtype(frame.dtype, np.signedinteger):
            raise TypeError("invalid_dtype")

        result: bool = self._vad.is_speech(frame)

        raw = result > 0
        if raw == self._run_value:
            self._run_length += 1
        else:
            self._run_value = raw
            self._run_length = 1

        if self._run_value != context.is_speech:
            if self._run_value and self._run_length >= self._rise_length:
                context.is_speech = True
                _LOG.info("vad: true")
            if not self._run_value and self._run_length >= self._fall_length:
                context.is_speech = False
                _LOG.info("vad: false")

    def reset(self) -> None:
        """ Resets the current state """
        self._run_value = 0
        self._run_length = 0

    def close(self) -> None:
        """ Close interface for use in pipeline """
        self.reset()


class VoiceActivityTrigger:
    """ Voice Activity Detector trigger pipeline component """

    def __init__(self) -> None:
        self._is_speech = False

    def __call__(self, context: SpeechContext, frame: np.ndarray) -> None:
        """Activates speech context whenever speech is detected

        Args:
            context (SpeechContext): State based information that needs to be shared
            between pieces of the pipeline
            frame (np.ndarray): Single frame of PCM-16 audio from an input source

        """
        if context.is_speech != self._is_speech:
            if context.is_speech:
                context.is_active = True
            self._is_speech = context.is_speech

    def close(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._is_speech = False
