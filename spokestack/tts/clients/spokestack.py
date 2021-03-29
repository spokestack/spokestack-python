"""
This module contains the Spokestack client for text to speech
"""
import base64
import hashlib
import hmac
import json
from typing import Any, Iterator

import requests

_MODES = {
    "ssml": "synthesizeSsml",
    "markdown": "synthesizeMarkdown",
    "text": "synthesizeText",
}


class TextToSpeechClient:
    """Spokestack Text to Speech Client

    Args:
        key_id (str): identity from spokestack api credentials
        key_secret (str): secret key from spokestack api credentials
        url (str): spokestack api url
    """

    def __init__(
        self, key_id: str, key_secret: str, url: str = "https://api.spokestack.io/v1"
    ) -> None:

        self._key_id = key_id
        self._key = key_secret.encode("utf-8")
        self._url = url

    def synthesize(
        self,
        utterance: str,
        mode: str = "text",
        voice: str = "demo-male",
        profile: str = "default",
    ) -> Iterator[bytes]:
        """Converts the given utterance to speech.

        Text can be formatted as plain text (`mode="text"`),
        SSML (`mode="ssml"`), or Speech Markdown (`mode="markdown"`).

        This method also supports different formats for the synthesized
        audio via the `profile` argument. The supported profiles and
        their associated formats are:

        - `default`: 24kHz, 64kbps mono MP3
        - `alexa`: 24kHz, 48kbps mono MP3
        - `discord`: 48kHz, 64kbpz stereo OPUS
        - `twilio`: 8kHz, 64kbpz mono MP3

        Args:
            utterance (str): string that needs to be rendered as speech.
            mode (str): synthesis mode to use with utterance. text, ssml, markdown.
            voice (str): name of the tts voice.
            profile (str): name of the audio profile used to create the
                           resulting stream.

        Returns:
            (Iterator[bytes]): Encoded audio response in the form of a sequence of bytes

        """
        audio_url = self.synthesize_url(utterance, mode, voice, profile)
        response = requests.get(audio_url, stream=True)

        if response.status_code != 200:
            raise Exception(response.reason)

        return response.iter_content(chunk_size=None)

    def synthesize_url(
        self,
        utterance: str,
        mode: str = "text",
        voice: str = "demo-male",
        profile: str = "default",
    ) -> str:
        """Converts the given utterance to speech accessible by a URL.

        Text can be formatted as plain text (`mode="text"`),
        SSML (`mode="ssml"`), or Speech Markdown (`mode="markdown"`).

        This method also supports different formats for the synthesized
        audio via the `profile` argument. The supported profiles and
        their associated formats are:

        - `default`: 24kHz, 64kbps mono MP3
        - `alexa`: 24kHz, 48kbps mono MP3
        - `discord`: 48kHz, 64kbpz stereo OPUS
        - `twilio`: 8kHz, 64kbpz mono MP3

        Args:
            utterance (str): string that needs to be rendered as speech.
            mode (str): synthesis mode to use with utterance. text, ssml, markdown.
            voice (str): name of the tts voice.
            profile (str): name of the audio profile used to create the
                           resulting stream.

        Returns: URL of the audio clip

        """
        body = self._build_body(utterance, mode, voice, profile)
        signature = base64.b64encode(
            hmac.new(self._key, body.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")
        headers = {
            "Authorization": f"Spokestack {self._key_id}:{signature}",
            "Content-Type": "application/json",
        }
        response: Any = requests.post(self._url, headers=headers, data=body)

        if response.status_code != 200:
            raise Exception(response.reason)

        response = response.json()
        if "errors" in response:
            raise TTSError(response["errors"])

        return response["data"][_MODES[mode]]["url"]

    @staticmethod
    def _build_body(message: str, mode: str, voice: str, profile: str) -> str:
        if mode not in _MODES:
            raise ValueError("invalid_mode")

        query = f"""
        query PythonSynthesis(
          $voice: String!, ${mode}: String!, $profile: SynthesisProfile) {{
            {_MODES[mode]}(voice: $voice, {mode}: ${mode}, profile: $profile) {{url}}
        }}
        """
        return json.dumps(
            {
                "query": query,
                "variables": {
                    "voice": voice,
                    mode: message,
                    "profile": profile.upper(),
                },
            }
        )


class TTSError(Exception):
    """ Text to speech error wrapper """

    def __init__(self, response: Any) -> None:
        messages = [error["message"] for error in response]
        super().__init__(messages)
