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
    "ssml": "synthesizeSSML",
    "markdown": "synthesizeMarkdown",
    "text": "synthesizeText",
}


class TextToSpeechClient:
    """ Spokestack Text to Speech Client

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
        self, utterance: str, mode: str = "text", voice: str = "demo-male",
    ) -> Iterator[bytes]:
        """ Converts the given utterance to speech

        Args:
            utterance (str): string that needs to be rendered as speech.
            mode (str): synthesis mode to use with utterance. text, ssml, markdown.
            voice (str): name of the tts voice.

        Returns:
            (Iterator[bytes]): Encoded audio response in the form of a sequence of bytes

        """
        audio_url = self.synthesize_url(utterance, mode, voice)
        response = requests.get(audio_url, stream=True)

        if response.status_code != 200:
            raise Exception(response.reason)

        return response.iter_content(chunk_size=None)

    def synthesize_url(
        self, utterance: str, mode: str = "text", voice: str = "demo-male",
    ) -> str:
        """ Converts the given uttrance to speech accessible by a URL.

        Args:
            utterance (str): string that needs to be rendered as speech.
            mode (str): synthesis mode to use with utterance. text, ssml, markdown.
            voice (str): name of the tts voice.

        Returns: URL of the audio clip

        """
        body = self._build_body(utterance, mode, voice)
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

    def _build_body(self, message, mode, voice):
        if mode == "ssml":
            return json.dumps(
                {
                    "query": """
               query synthesis($voice: String!, $ssml: String!) {
                 synthesizeSSML(voice: $voice, ssml: $ssml) {url}
               }
               """,
                    "variables": {"voice": voice, "ssml": message},
                }
            )
        elif mode == "markdown":
            return json.dumps(
                {
                    "query": """
                           query synthesis($voice: String!, $markdown: String!) {
                             synthesizeMarkdown(voice: $voice, markdown: $markdown) {
                             url}
                           }
                           """,
                    "variables": {
                        "voice": voice,
                        "markdown": message,
                        "method": "synthesizeMarkdown",
                    },
                }
            )
        elif mode == "text":
            return json.dumps(
                {
                    "query": """
                           query synthesis($voice: String!, $text: String!) {
                             synthesizeText(voice: $voice, text: $text) {url}
                           }
                           """,
                    "variables": {"voice": voice, "text": message},
                }
            )
        else:
            raise ValueError("invalid_mode")


class TTSError(Exception):
    """ Text to speech error wrapper """

    def __init__(self, response) -> None:
        messages = [error["message"] for error in response]
        super().__init__(messages)
