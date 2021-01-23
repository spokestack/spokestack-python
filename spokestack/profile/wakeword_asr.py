"""
Pipeline profile for pyaudio input, vad, wakeword, and asr
"""
from typing import Any

from spokestack.activation_timeout import ActivationTimeout
from spokestack.agc.webrtc import AutomaticGainControl
from spokestack.asr.spokestack.speech_recognizer import CloudSpeechRecognizer
from spokestack.io.pyaudio import PyAudioInput
from spokestack.nsx.webrtc import AutomaticNoiseSuppression
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector
from spokestack.wakeword.tflite import WakewordTrigger


class WakewordSpokestackASR:
    """ TFLite wakeword with Spokestack speech recognition. """

    @staticmethod
    def create(
        spokestack_id: str,
        spokestack_secret: str,
        sample_rate: int = 16000,
        frame_width: int = 20,
        model_dir: str = "",
        **kwargs: Any,
    ) -> SpeechPipeline:
        """Creates a speech pipeline instance from profile

        Args:
            spokestack_id (str): spokestack API id.
            spokestack_secret (str): Spokestack API secret.
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
                WakewordTrigger(model_dir=model_dir, **kwargs),
                ActivationTimeout(frame_width=frame_width, **kwargs),
                CloudSpeechRecognizer(
                    spokestack_secret=spokestack_secret,
                    spokestack_id=spokestack_id,
                    **kwargs,
                ),
            ],
        )
        return pipeline
