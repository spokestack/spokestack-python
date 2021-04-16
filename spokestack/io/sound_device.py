"""Audio input component based on the sounddevice library."""
from typing import Any

import numpy as np
import sounddevice as sd


class SoundDeviceInput:
    """
    Audio input source using sounddevice.

    Parameters
    ----------
    sample_rate (Hz): int, optional
        audio sample rate, by default 16000
    frame_width (ms): int, optional
        size of the audio frame, by default 20
    """

    def __init__(
        self, sample_rate: int = 16000, frame_width: int = 20, **kwargs: Any
    ) -> None:

        self._frame_size = int(sample_rate / 1000 * frame_width)
        self._sample_rate = sample_rate
        self._stream = sd.Stream(samplerate=sample_rate)

    def read(self) -> np.array:
        """
        Read audio from input source.

        Returns
        -------
        np.array
            NumPy array of audio
        """
        return self._stream.read(self._frame_size)

    def start(self) -> None:
        """Start the audio stream."""
        self._stream.start()

    def stop(self) -> None:
        """Stop the audio stream."""
        self._stream.stop()

    def close(self) -> None:
        """Close the audio stream."""
        self._stream.close()

    @property
    def is_active(self) -> bool:
        """
        Active status of the audio stream.

        Returns
        -------
        bool
            True if active, False otherwise
        """
        return self._stream.active

    @property
    def is_stopped(self) -> bool:
        """
        Stop status of the audio stream.

        Returns
        -------
        bool
            True if stopped, False otherwise
        """
        return self._stream.stopped
