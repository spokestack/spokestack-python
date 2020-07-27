"""
This module contains the webrtc component for
voice activity detection (vad)
"""

import webrtcvad  # type: ignore

from spokestack.context import SpeechContext


MODE = {"quality": 0, "low_bitrate": 1, "aggresive": 2, "very_aggresive": 3}


class VoiceActivityDetector:
    """ This class detects the presence of voice in a frame of audio.

    :param sample_rate: sample rate of the audio
    :type sample_rate: int
    :param frame_width: width of the audio frame
    :type frame_width: int
    :param vad_rise_delay: rise delay
    :type vad_rise_delay: int
    :param vad_fall_delay: fall delay
    :type vad_fall_delay: int
    :param mode: mode for vad
    :type mode: str
    """

    def __init__(
        self,
        sample_rate: int,
        frame_width: int,
        vad_rise_delay: int,
        vad_fall_delay: int,
        mode: str,
    ) -> None:

        self.sample_rate = sample_rate
        self.rise_length = vad_rise_delay // frame_width
        self.fall_length = vad_fall_delay // frame_width

        self.vad = webrtcvad.Vad(MODE[mode])

        self.run_value = 0
        self.run_length = 0

    def __call__(self, context: SpeechContext, frame: bytes) -> None:
        """ Processes a single frame of audio to detemine if voice is present

        :param context: State based information that needs
        to be shared between pieces of the pipeline
        :type context: SpeechContext
        :param frame: Single frame of audio from an input source
        :type frame: bytes
        """
        result: bool = self.vad.is_speech(frame, self.sample_rate)

        raw = int(result) > 0
        if raw == self.run_value:
            self.run_length += 1
        else:
            self.run_value = raw
            self.run_length = 1

        if self.run_value != context.is_speech:
            if self.run_value and self.run_length >= self.rise_length:
                context.is_speech = True
            if not self.run_value and self.run_length >= self.fall_length:
                context.is_speech = False

    def reset(self) -> None:
        """ Resets the current state """
        self.run_value = 0
        self.run_length = 0
