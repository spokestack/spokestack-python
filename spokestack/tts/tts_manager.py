"""
This module contains the Spokestack text to speech manager which handles a
text to speech client, decodes the returned audio, and writes the audio to
the specified output
"""
from streamp3 import MP3Decoder  # type: ignore


class TextToSpeechManager:
    """ Manages tts client and outputs target

        Args:
            client: Text to speech client that returns encoded mp3 audio
            output: Audio outputs target
    """

    def __init__(self, client, output) -> None:
        self._client = client
        self._output = output

    def synthesize(self, **kwargs) -> None:
        """ Synthesizes speech and writes to output

        Args:
            **kwargs: keyword arguments passed to the respective tts clients

        """
        encoded = self._client.synthesize_speech(**kwargs)
        decoded = MP3Decoder(encoded)
        self._output.write(decoded)

    def close(self) -> None:
        """ closes the client and output """
        self._client = None
        self._output = None
