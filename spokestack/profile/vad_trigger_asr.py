"""
Pipeline profile with vad trigger and asr
"""
from spokestack.activation_timeout import ActivationTimeout
from spokestack.asr.speech_recognizer import CloudSpeechRecognizer
from spokestack.io.pyaudio import PyAudioInput
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector, VoiceActivityTrigger


class VoiceActivityTriggerASR:
    """ VAD Trigger ASR

    Args:
            spokestack_id (str):
            spokestack_secret (str):
            sample_rate (int):
            frame_width (int):
            vad_rise_delay (int):
            vad_fall_delay (int):
            idle_timeout (int):
            mode (int):

    """

    def __init__(
        self,
        spokestack_id: str,
        spokestack_secret: str,
        sample_rate: int = 16000,
        frame_width: int = 20,
        **kwargs
    ) -> None:
        self._pipeline = SpeechPipeline(
            input_source=PyAudioInput(
                sample_rate=sample_rate, frame_width=frame_width, **kwargs
            ),
            stages=[
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

    def run(self) -> None:
        """ Runs the configured pipeline. """
        self._pipeline.run()

    def start(self) -> None:
        self._pipeline.start()
