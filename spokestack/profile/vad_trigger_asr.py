"""
Pipeline profile with vad trigger and asr
"""
from typing import Any

from spokestack.activation_timeout import ActivationTimeout
from spokestack.agc.webrtc import AutomaticGainControl
from spokestack.asr.spokestack.speech_recognizer import CloudSpeechRecognizer
from spokestack.io.pyaudio import PyAudioInput
from spokestack.nsx.webrtc import AutomaticNoiseSuppression
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector, VoiceActivityTrigger


class VoiceActivityTriggerSpokestackASR:
    """ VAD Trigger ASR """

    @staticmethod
    def create(
        spokestack_id: str,
        spokestack_secret: str,
        sample_rate: int = 16000,
        frame_width: int = 20,
        **kwargs: Any
    ) -> SpeechPipeline:
        """

        Args:
            spokestack_id (str): spokestack API id.
            spokestack_secret (str): Spokestack API secret.
            sample_rate (int): sample rate of the audio (Hz).
            frame_width (int): width of the audio frame: 10, 20, or 30 (ms).

        Returns:
            SpeechPipeline instance with profile configuration.

        """
        pipeline = SpeechPipeline(
            input_source=PyAudioInput(
                sample_rate=sample_rate, frame_width=frame_width, **kwargs
            ),
            stages=[
                AutomaticGainControl(sample_rate=sample_rate, frame_width=frame_width),
                AutomaticNoiseSuppression(sample_rate=sample_rate),
                VoiceActivityDetector(
                    sample_rate=sample_rate, frame_width=frame_width, **kwargs
                ),
                VoiceActivityTrigger(),
                ActivationTimeout(frame_width=frame_width, **kwargs),
                CloudSpeechRecognizer(
                    spokestack_id=spokestack_id,
                    spokestack_secret=spokestack_secret,
                    sample_rate=sample_rate,
                    frame_width=frame_width,
                    **kwargs
                ),
            ],
        )
        return pipeline
