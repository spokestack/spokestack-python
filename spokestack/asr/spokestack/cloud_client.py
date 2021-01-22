"""
This module contains the websocket logic used to communicate with
Spokestack's cloud-based ASR service.
"""
import base64
import hashlib
import hmac
import json
from typing import Any, Dict, List, Union

import numpy as np
from websocket import WebSocket


class CloudClient:
    """Spokestack client for cloud based speech to text

    Args:
        key_id (str): identity from spokestack api credentials
        key_secret (str): secret key from spokestack api credentials
        socket_url (str): url for socket connection
        audio_format (str): format of input audio
        sample_rate (int): audio sample rate (kHz)
        language (str): language for recognition
        limit (int): Limit of messages per api response
        idle_timeout (Any): Time before client timeout. Defaults to None
    """

    def __init__(
        self,
        key_id: str,
        key_secret: str,
        socket_url: str = "wss://api.spokestack.io",
        audio_format: str = "PCM16LE",
        sample_rate: int = 16000,
        language: str = "en",
        limit: int = 10,
        idle_timeout: Union[float, None] = None,
    ) -> None:

        self._body: str = json.dumps(
            {
                "format": audio_format,
                "rate": sample_rate,
                "language": language,
                "limit": limit,
            }
        )
        self._socket_url: str = socket_url
        self._key_id: str = key_id
        self._key: bytes = key_secret.encode("utf-8")
        signature = hmac.new(
            self._key, self._body.encode("utf-8"), hashlib.sha256
        ).digest()
        self._signature = base64.b64encode(signature).decode("utf-8")
        self._socket: Any = None

        self._response: Dict[str, Any] = {
            "error": None,
            "final": True,
            "hypotheses": [],
            "status": None,
        }
        self._sample_rate: int = sample_rate
        self._idle_timeout = idle_timeout
        self._idle_count: int = 0

    def __call__(self, audio: Union[bytes, np.ndarray], limit: int = 1) -> List[str]:
        """Audio to text interface for the cloud client

        Args:
            audio (bytes|np.ndarray): input audio can be in the form of
                                      bytes or np.float, np.int16 array with
                                      conversions handled. other types with produce
                                      a TypeError
            limit (int): number of predictions to return

        Returns: list of transcripts, and their confidence values of size limit

        """
        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, np.int16)
        elif np.issubdtype(audio.dtype, np.floating):
            # convert and rescale to PCM-16
            audio = (audio * (2 ** 15 - 1)).astype(np.int16)
        elif not np.issubdtype(audio.dtype, np.int16):
            raise TypeError("invalid_audio")

        chunk_size = self._sample_rate
        self.connect()
        self.initialize()

        for i in range(0, len(audio), chunk_size):
            frame = audio[i:][:chunk_size]
            self.send(frame)
            self.receive()

        self.end()
        while not self._response["final"]:
            self.receive()
        self.disconnect()

        hypotheses = self._response.get("hypotheses", [])
        return hypotheses[:limit]

    @property
    def is_connected(self) -> bool:
        """ status of the socket connection """
        if self._socket:
            return True
        return False

    def connect(self) -> None:
        """ connects to websocket """
        if self._socket is None:
            self._socket = WebSocket()
            self._socket.connect(f"{self._socket_url}/v1/asr/websocket")

    def initialize(self) -> None:
        """ sends/receives the initial api request """
        if not self._socket:
            raise ConnectionError("Not Connected")

        message = {
            "keyId": self._key_id,
            "signature": self._signature,
            "body": self._body,
        }
        self._socket.send(json.dumps(message))
        self._response = json.loads(self._socket.recv())
        if not self._response["status"] == "ok":
            raise APIError(self._response)

    def disconnect(self) -> None:
        """ disconnects client socket connection """
        if self._socket:
            self._socket.close()
            self._socket = None

    def send(self, frame: np.ndarray) -> None:
        """sends a single frame of audio

        Args:
            frame (np.ndarray): segment of PCM-16 encoded audio

        """
        if self._socket:
            self._socket.send_binary(frame.tobytes())
        else:
            raise ConnectionError("Not Connected")

    def end(self) -> None:
        """ sends empty string in binary to indicate last frame """
        if self._socket:
            self._socket.send_binary(b"")
        else:
            raise ConnectionError("Not Connected")

    def receive(self) -> None:
        """ receives the api response """
        if self._socket:
            timeout = self._socket.timeout
            try:
                self._socket.timeout = 0
                response = self._socket.recv()
                self._response = json.loads(response)
            except Exception:
                pass
            self._socket.timeout = timeout
        else:
            raise ConnectionError("Not Connected")

    @property
    def response(self) -> dict:
        """ current response message"""
        return self._response

    @property
    def is_final(self) -> bool:
        """ status of most recent sever response """
        return self._response["final"]

    @property
    def idle_timeout(self) -> Any:
        """ property for maximum idle time """
        return self._idle_timeout

    @property
    def idle_count(self) -> int:
        """ current counter of idle time """
        return self._idle_count

    @idle_count.setter
    def idle_count(self, value: int) -> None:
        """ sets the idle counter"""
        self._idle_count = value


class APIError(Exception):
    """Spokestack api error pass through

    Args:
        response (dict): message from the api service
    """

    def __init__(self, response: dict) -> None:
        super().__init__(response["error"])
