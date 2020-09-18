"""
This module contains the Spokestack client for text to speech
"""
import base64
import hashlib
import hmac
import json

import requests


MODES = {
    "ssml": "synthesizeSSML",
    "markdown": "synthesizeMarkdown",
    "text": "synthesizeText",
}


class TextToSpeechClient:
    """ Spokestack Text to Speech Client

    Args:
        key_id (str):
        key_secret (str):
        url (str):
    """

    def __init__(self, key_id: str, key_secret: str, url: str) -> None:

        self._key_id = key_id
        self._key = key_secret.encode("utf-8")
        self._url = url

    def synthesize_speech(
        self, utterance: str, mode: str = "text", voice: str = "demo-male"
    ) -> bytes:
        """ Converts the given utterance to speech

        Args:
            utterance (str):
            mode (str):
            voice (str):

        Returns: Encoded Audio

        """
        body = self._build_body(utterance, mode, voice)
        signature = base64.b64encode(
            hmac.new(self._key, body.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")
        headers = {
            "Authorization": f"Spokestack {self._key_id}:{signature}",
            "Content-Type": "application/json",
        }

        response = requests.post(self._url, headers=headers, data=body).json()

        if "errors" in response:
            raise TTSError(response["errors"])

        response = requests.get(response["data"][MODES[mode]]["url"])

        return response.content

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
