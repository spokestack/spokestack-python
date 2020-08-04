"""
This module uses pyaudio to receive input from a
device microphone
"""
import pyaudio  # type: ignore


class PyAudioMicrophoneInput:
    """This class retrieves audio from an input device

    Args:
        sample_rate (int): desired sample rate for input
        frame_width (int): desired frame width for input
        exception_on_overflow (bool): produce exception for input overflow
    """

    def __init__(
        self, sample_rate: int, frame_width: int, exception_on_overflow: bool = True
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate / 1000 * frame_width)
        self._exception_on_overflow = exception_on_overflow
        self._audio = pyaudio.PyAudio()
        device = self._audio.get_default_input_device_info()
        self._stream = self._audio.open(
            input=True,
            input_device_index=device["index"],
            format=pyaudio.paInt16,
            frames_per_buffer=self._frame_size,
            channels=1,
            rate=self._sample_rate,
            start=False,
        )

    def read(self) -> bytes:
        """Reads a single frame of audio

        Returns:
            bytes: single frame of audio
        """
        return self._stream.read(
            self._frame_size, exception_on_overflow=self._exception_on_overflow
        )

    def start(self) -> None:
        """ Starts the audio stream """
        self._stream.start_stream()

    def stop(self) -> None:
        """ Stops the audio stream """
        self._stream.stop_stream()

    def close(self) -> None:
        """ Closes the audio stream """
        self._stream.close()

    @property
    def is_active(self) -> bool:
        """Stream active property

        Returns:
            bool: 'True' if stream is active, 'False' otherwise
        """
        return self._stream.is_active()

    @property
    def is_stopped(self) -> bool:
        """Stream stopped property

        Returns:
            bool: 'True' if stream is stopped, 'False' otherwise
        """
        return self._stream.is_stopped()
