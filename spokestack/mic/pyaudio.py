"""
This module uses pyaudio to receive input from a
device microphone
"""
import pyaudio  # type: ignore

from spokestack.config import SpeechConfig


class PyAudioMicrophoneInput:
    """
    Retrieve microphone input with PyAudio

    Args:
        config (SpeechConfig): configuration paramaters
    """

    def __init__(self, config: SpeechConfig) -> None:

        self.config = config
        self.frame_size = int(config.sample_rate / 1000 * config.frame_width)
        self.audio = pyaudio.PyAudio()

    def build(self) -> None:
        device = self.audio.get_default_input_device_info()
        self._stream = self.audio.open(
            input=True,
            input_device_index=device["index"],
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.sample_rate,
            start=False,
        )

    def read(self) -> bytes:
        return self._stream.read(self.frame_size, exception_on_overflow=False)

    def start(self) -> None:
        self._stream.start_stream()

    def stop(self) -> None:
        self._stream.stop_stream()

    def is_active(self) -> bool:
        return self._stream.is_active()

    def is_stopped(self) -> bool:
        return self._stream.is_stopped()

    def close(self) -> None:
        self._stream.close()
