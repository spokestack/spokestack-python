"""
Pipeline profile for pyaudio input, vad, wakeword, and asr
"""
from spokestack.activation_timeout import ActivationTimeout
from spokestack.asr.speech_recognizer import CloudSpeechRecognizer
from spokestack.io.pyaudio import PyAudioInput
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector
from spokestack.wakeword.tflite import WakewordTrigger


class WakewordASR:
    """ Wakeword ASR Profile """

    def __init__(
        self,
        spokestack_id: str,
        spokestack_secret: str,
        sample_rate: int = 16000,
        frame_width: int = 20,
        model_dir: str = "",
        **kwargs,
    ) -> None:
        self._pipeline = SpeechPipeline(
            input_source=PyAudioInput(
                frame_width=frame_width, sample_rate=sample_rate, **kwargs
            ),
            stages=[
                VoiceActivityDetector(
                    frame_width=frame_width, sample_rate=sample_rate, **kwargs,
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

    def run(self) -> None:
        """ Runs the configured pipeline. """
        self._pipeline.run()

    def start(self) -> None:
        self._pipeline.start()
