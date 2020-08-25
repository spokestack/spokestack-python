"""
This module contains the websocket logic used to communicate with
Spokestack's cloud-based ASR service.
"""
import base64
import hashlib
import hmac
import json

from websocket import WebSocket  # type: ignore


class CloudClient:
    """ spokestack cloud client """

    def __init__(
        self,
        socket_url: str,
        key_id: str,
        key_secret: str,
        audio_format: str = "PCM16LE",
        sample_rate: int = 16000,
        language: str = "en",
        limit: int = 10,
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
        self._socket: WebSocket = WebSocket()
        self._response: dict = {
            "error": None,
            "final": False,
            "hypotheses": [],
            "status": None,
        }
        self._connected = False
        self._is_final = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._socket.close()

    def connect(self) -> None:
        if self._connected:
            raise ConnectionError("Already Connected")
        self._socket.connect(f"{self._socket_url}/v1/asr/websocket")
        self._connected = True

    def initialize(self) -> None:
        if not self._connected:
            raise ConnectionError("Not Connected")

        message = {
            "keyId": self._key_id,
            "signature": self._signature,
            "body": self._body,
        }
        self._socket.send(json.dumps(message))
        self._socket.recv()

    def disconnect(self) -> None:
        if self._connected:
            self._socket.close()
        self._connected = False

    def send(self, frame):
        self._socket.send_binary(frame)

    def end(self):
        # empty string to indicate end
        self._socket.send_binary(b"")

    def receive(self):
        timeout = self._socket.timeout
        try:
            self._socket.timeout = 0
            response = json.loads(self._socket.recv())
            self._response = response
            self._is_final = self._response["final"]
        except Exception:
            pass
        self._socket.timeout = timeout

    @property
    def response(self) -> dict:
        return self._response

    @property
    def is_final(self):
        return self._is_final
