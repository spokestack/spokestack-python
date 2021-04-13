"""
Pipeline profile for pyaudio input, vad, keyword.
"""
from typing import Any, List

from spokestack.activation_timeout import ActivationTimeout
from spokestack.agc.webrtc import AutomaticGainControl
from spokestack.asr.keyword.tflite import KeywordRecognizer
from spokestack.io.pyaudio import PyAudioInput
from spokestack.nsx.webrtc import AutomaticNoiseSuppression
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector, VoiceActivityTrigger


class SpokestackKeyword:
    """Spokestack keyword profile."""

    @staticmethod
    def create(
        classes: List[str],
        model_dir: str,
        sample_rate: int = 16000,
        frame_width: int = 20,
        **kwargs: Any
    ) -> SpeechPipeline:
        """Create a speech pipeline instance from profile.

        Args:
            model_dir (str): Directory containing the tflite keyword models.
            classes: (List(str)): Classes for the keyword model to recognize
            sample_rate (int): sample rate of the audio (Hz).
            frame_width (int): width of the audio frame: 10, 20, or 30 (ms).

        """
        pipeline = SpeechPipeline(
            input_source=PyAudioInput(
                frame_width=frame_width, sample_rate=sample_rate, **kwargs
            ),
            stages=[
                AutomaticGainControl(sample_rate=sample_rate, frame_width=frame_width),
                AutomaticNoiseSuppression(sample_rate=sample_rate),
                VoiceActivityDetector(
                    sample_rate=sample_rate, frame_width=frame_width, **kwargs
                ),
                VoiceActivityTrigger(),
                KeywordRecognizer(
                    classes=classes,
                    model_dir=model_dir,
                    sample_rate=sample_rate,
                    **kwargs,
                ),
                ActivationTimeout(frame_width=frame_width, **kwargs),
            ],
        )
        return pipeline
