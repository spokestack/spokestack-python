"""
This module uses pyaudio to receive input from a
device microphone
"""
import pyaudio  # type: ignore


class PyAudioMicrophoneInput:
    """ This class retrieves audio from an input device

    :param sample_rate: desired sample rate for input
    :type sample_rate: int
    :param frame_width: desired frame width for input
    :type frame_width: int
    :param exception_on_overflow: produce exception for input overflow
    :type exception_on_overflow: bool
    """

    def __init__(
        self, sample_rate: int, frame_width: int, exception_on_overflow: bool = True
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_size = int(sample_rate / 1000 * frame_width)
        self._exeception_on_overflow = exception_on_overflow
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
        """ Reads a single frame of audio

        :return: single frame of audio
        :rtype: bytes
        """
        return self._stream.read(
            self._frame_size, exception_on_overflow=self._exeception_on_overflow
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
        """ Stream active property

        :return: 'True' if stream is active, 'False' otherwise
        :rtype: bool
        """
        return self._stream.is_active()

    @property
    def is_stopped(self) -> bool:
        """ Stream stopped property

        :return: 'True' if stream is stopped, 'False' otherwise
        :rtype: bool
        """
        return self._stream.is_stopped()
