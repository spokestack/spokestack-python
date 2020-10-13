"""
This module contains the Spokestack text to speech manager which handles a
text to speech client, decodes the returned audio, and writes the audio to
the specified output.
"""


class TextToSpeechManager:
    """ Manages tts client and io target.

        Args:
            client: Text to speech client that returns encoded mp3 audio
            output: Class that handles the TTS response
    """

    def __init__(self, client, output) -> None:
        self._client = client
        self._output = output

    def synthesize(self, utterance: str, mode: str, voice: str) -> None:
        """ Synthesizes the given utterance with the voice and format provided.

        Args:
            utterance (str): string that needs to be rendered as speech.
            mode (str): synthesis mode to use with utterance. text, ssml, markdown, etc.
            voice (str): name of the tts voice.

        """
        response = self._client.synthesize(utterance, mode, voice)
        self._output.write(response)

    def close(self) -> None:
        """ Closes the client and output. """
        self._client = None
        self._output = None
