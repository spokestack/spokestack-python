"""
This module contains the websocket logic used to communicate with
Spokestack's cloud-based ASR service.
"""
import base64
import hashlib
import hmac
import json
from typing import Any, Dict, List

import numpy as np  # type: ignore
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
        idle_timeout: Any = None,
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
            "final": False,
            "hypotheses": [],
            "status": None,
        }
        self._sample_rate: int = sample_rate
        self._is_final: bool = False
        self._idle_timeout = idle_timeout
        self._idle_count: int = 0

    def __call__(self, audio: np.ndarray, n_best: int = 1) -> List[str]:
        """ Audio to text interface for the cloud client

        Args:
            audio (np.ndarray): np.int16 array of audio
            n_best (int): number of predictions to return

        Returns: n_best list of transcripts

        """

        chunk_size = 2 * int(0.01 * self._sample_rate)
        self.connect()
        self.initialize()

        for i in range(0, len(audio), chunk_size):
            frame = audio[i:][:chunk_size]
            self.send(frame)
            self.receive()

        self.end()
        while not self._is_final:
            if self._idle_timeout and (self._idle_count > self._idle_timeout):
                break
            else:
                self.receive()
                self._idle_count += 1
        self.disconnect()

        transcripts = []
        hypotheses = self._response.get("hypotheses")
        if hypotheses:
            for hypothesis in hypotheses[:n_best]:
                transcripts.append(hypothesis.get("transcript"))

        return transcripts

    @property
    def is_connected(self) -> bool:
        if self._socket:
            return True
        return False

    def close(self) -> None:
        self._socket.close()

    def connect(self) -> None:
        if self._socket:
            pass
        else:
            self._socket = WebSocket()
            self._socket.connect(f"{self._socket_url}/v1/asr/websocket")

    def initialize(self) -> None:
        if not self._socket:
            raise ConnectionError("Not Connected")

        message = {
            "keyId": self._key_id,
            "signature": self._signature,
            "body": self._body,
        }
        self._socket.send(json.dumps(message))
        self._response = json.loads(self._socket.recv())
        self._is_final = False
        if not self._response["status"] == "ok":
            raise APIError(self._response)

    def disconnect(self) -> None:
        if self._socket:
            self.close()
        self._socket = None

    def send(self, frame):
        if self._socket:
            self._socket.send_binary(frame)
        else:
            raise ConnectionError("Not Connected")

    def end(self):
        if self._socket:
            # empty string to indicate last frame
            self._socket.send_binary(b"")
        else:
            raise ConnectionError("Not Connected")

    def receive(self):
        if self._socket:
            timeout = self._socket.timeout
            try:
                self._socket.timeout = 0
                response = self._socket.recv()
                self._response = json.loads(response)
                self._is_final = self._response["final"]
            except Exception:
                pass
            self._socket.timeout = timeout
        else:
            raise ConnectionError("Not Connected")

    @property
    def response(self) -> dict:
        return self._response

    @property
    def is_final(self) -> bool:
        return self._is_final

    @property
    def idle_timeout(self) -> Any:
        return self._idle_timeout

    @property
    def idle_count(self) -> int:
        return self._idle_count

    @idle_count.setter
    def idle_count(self, value: int):
        self._idle_count = value


class APIError(Exception):
    """ Spokestack """

    def __init__(self, response) -> None:
        super().__init__(response["error"])
