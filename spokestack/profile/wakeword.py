"""
Pipeline profile for pyaudio input, vad, wakeword.
"""
from typing import Any

from spokestack.activation_timeout import ActivationTimeout
from spokestack.agc.webrtc import AutomaticGainControl
from spokestack.io.pyaudio import PyAudioInput
from spokestack.nsx.webrtc import AutomaticNoiseSuppression
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector
from spokestack.wakeword.tflite import WakewordTrigger


class SpokestackWakeword:
    """ Spokestack wakeword profile. """

    @staticmethod
    def create(
        model_dir: str, sample_rate: int = 16000, frame_width: int = 20, **kwargs: Any
    ) -> SpeechPipeline:
        """Creates a speech pipeline instance from profile

        Args:
            sample_rate (int): sample rate of the audio (Hz).
            frame_width (int): width of the audio frame: 10, 20, or 30 (ms).
            model_dir (str): Directory containing the tflite wakeword models.

        Returns:

        """
        pipeline = SpeechPipeline(
            input_source=PyAudioInput(
                frame_width=frame_width, sample_rate=sample_rate, **kwargs
            ),
            stages=[
                AutomaticGainControl(sample_rate=sample_rate, frame_width=frame_width),
                AutomaticNoiseSuppression(sample_rate=sample_rate),
                VoiceActivityDetector(
                    frame_width=frame_width,
                    sample_rate=sample_rate,
                    **kwargs,
                ),
                WakewordTrigger(
                    sample_rate=sample_rate,
                    frame_width=frame_width,
                    model_dir=model_dir,
                    **kwargs,
                ),
                ActivationTimeout(frame_width=frame_width, **kwargs),
            ],
        )
        return pipeline
