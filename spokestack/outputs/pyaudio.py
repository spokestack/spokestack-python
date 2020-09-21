"""
This module contains the class for using speaker outputs with pyaudio
"""
import pyaudio  # type: ignore


class PyAudioOutput:
    """ Outputs audio to the default system output

    Args:
        num_channels (int): number of audio channels
        sample_rate (int): sample rate of the audio (kHz)
    """

    def __init__(self, num_channels: int = 1, sample_rate: int = 24000) -> None:

        self._audio = pyaudio.PyAudio()
        self._device = self._audio.get_default_output_device_info()
        self._output = self._audio.open(
            output=True,
            input_device_index=self._device["index"],
            format=pyaudio.paInt16,
            channels=num_channels,
            rate=sample_rate,
        )

    def write(self, audio: bytes) -> None:
        """ Writes the passed audio to the system default outputs

        Args:
            audio (bytes): PCM-16 audio as bytes

        """
        for chunk in audio:
            self._output.write(chunk)
