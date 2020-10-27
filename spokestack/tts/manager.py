"""
This module contains the Spokestack text to speech manager which handles a
text to speech client, decodes the returned audio, and writes the audio to
the specified output.
"""
from streamp3 import MP3Decoder  # type: ignore


class TextToSpeechManager:
    """Manages tts client and io target.

    Args:
        client: Text to speech client that returns encoded mp3 audio
        output: Audio io target
    """

    def __init__(self, client, output) -> None:
        self._client = client
        self._output = output

    def synthesize(self, utterance: str, mode: str, voice: str) -> None:
        """Synthesizes the given utterance with the voice and format provided.

        Args:
            utterance (str): string that needs to be rendered as speech.
            mode (str): synthesis mode to use with utterance. text, ssml, markdown, etc.
            voice (str): name of the tts voice.

        """
        stream = self._client.synthesize(utterance, mode, voice)
        stream = SequenceIO(stream)
        for frame in MP3Decoder(stream):
            self._output.write(frame)

    def close(self) -> None:
        """ Closes the client and output. """
        self._client = None
        self._output = None


class SequenceIO:
    """ Wrapper that allows for incrementally received audio to be decoded. """

    def __init__(self, sequence):
        self._sequence = iter(sequence)

    def read(self, size=None):
        try:
            return next(self._sequence)
        except StopIteration:
            return b""
