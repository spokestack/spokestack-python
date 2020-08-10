"""
This module contains the webrtc component for
voice activity detection (vad)
"""

import webrtcvad  # type: ignore

from spokestack.context import SpeechContext


QUALITY = 0
LOW_BITRATE = 1
AGGRESSIVE = 2
VERY_AGGRESSIVE = 3


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
        sample_rate: int,
        frame_width: int,
        vad_rise_delay: int,
        vad_fall_delay: int,
        mode: int = QUALITY,
    ) -> None:

        self._sample_rate: int = sample_rate
        self._rise_length: int = vad_rise_delay // frame_width
        self._fall_length: int = vad_fall_delay // frame_width

        self._vad = webrtcvad.Vad(mode)

        self._run_value: int = 0
        self._run_length: int = 0

    def __call__(self, context: SpeechContext, frame: bytes) -> None:
        """Processes a single frame of audio to detemine if voice is present

        Args:
            context (SpeechContext): State based information that needs to be shared
            between pieces of the pipeline
            frame (bytes): Single frame of audio from an input source

        """
        result: bool = self._vad.is_speech(frame, self._sample_rate)

        raw = result > 0
        if raw == self._run_value:
            self._run_length += 1
        else:
            self._run_value = raw
            self._run_length = 1

        if self._run_value != context.is_speech:
            if self._run_value and self._run_length >= self._rise_length:
                context.is_speech = True
            if not self._run_value and self._run_length >= self._fall_length:
                context.is_speech = False

    def reset(self) -> None:
        """ Resets the current state """
        self._run_value = 0
        self._run_length = 0
