"""
This module uses pyaudio for input and output processing
"""
from typing import Any

import numpy as np
import pyaudio


class PyAudioInput:
    """This class retrieves audio from an input device

    Args:
        sample_rate (int): desired sample rate for input (Hz)
        frame_width (int): desired frame width for input (ms)
        exception_on_overflow (bool): produce exception for input overflow
    """

    def __init__(
        self,
        sample_rate: int,
        frame_width: int,
        exception_on_overflow: bool = True,
        **kwargs: Any
    ) -> None:
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
            rate=sample_rate,
            start=False,
        )

    def read(self) -> np.array:
        """Reads a single frame of audio

        Returns:
            np.ndarray: single frame of PCM-16 audio
        """
        frame = self._stream.read(
            self._frame_size, exception_on_overflow=self._exception_on_overflow
        )
        return np.frombuffer(frame, np.int16)

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


class PyAudioOutput:
    """Outputs audio to the default system output

    Args:
        num_channels (int): number of audio channels
        sample_rate (int): sample rate of the audio (Hz)
    """

    def __init__(self, num_channels: int = 1, sample_rate: int = 24000) -> None:
        audio = pyaudio.PyAudio()
        device = audio.get_default_output_device_info()
        self._output = audio.open(
            output=True,
            input_device_index=device["index"],
            format=pyaudio.paInt16,
            channels=num_channels,
            rate=sample_rate,
        )

    def write(self, frame: bytes) -> None:
        """Writes a single frame of audio to output

        Args:
            frame (bytes): a single frame of audio

        """
        self._output.write(frame)
