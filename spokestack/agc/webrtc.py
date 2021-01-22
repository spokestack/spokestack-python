"""
This module contains the class for webrtc's automatic gain control
"""
from typing import Any

import numpy as np

from spokestack.context import SpeechContext
from spokestack.extensions.webrtc.agc import WebRtcAgc


class AutomaticGainControl:
    """WebRTC Automatic Gain Control

    Args:
            sample_rate (int): audio sample_rate. (Hz)
            frame_width (int): audio frame width. (Ms)
            target_level_dbfs (int): target peak audio level. (dBFS)
            compression_gain_db (int): dynamic range compression rate. (dB)
            limit_enable (bool): enables limiter in compression.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_width: int = 20,
        target_level_dbfs: int = 3,
        compression_gain_db: int = 15,
        limit_enable: bool = True,
        **kwargs: Any
    ) -> None:
        # validate sample rate
        self._sample_rate = sample_rate
        if self._sample_rate not in {8000, 16000, 32000}:
            raise ValueError("invalid_sample_rate")
        self._frame_width = frame_width
        # validate frame width
        if self._frame_width not in {10, 20}:
            raise ValueError("invalid_frame_width")

        self._agc = WebRtcAgc(
            sample_rate=self._sample_rate,
            frame_width=self._frame_width,
            target_level_dbfs=target_level_dbfs,
            compression_gain_db=compression_gain_db,
            limit_enable=limit_enable,
        )

    def __call__(self, context: SpeechContext, frame: np.array) -> None:
        """Main Entry Point

        Args:
            context (SpeechContext): State based information that needs to be shared
            between pieces of the pipeline
            frame (np.array): PCM-16 audio.
        """
        # validate frame size
        if len(frame) != self._sample_rate * self._frame_width // 1000:
            raise ValueError("invalid_frame_size")
        # validate dtype
        if not np.issubdtype(frame.dtype, np.signedinteger):
            raise TypeError("invalid_dtype")
        # run automatic gain control on the frame
        self._agc(frame)

    def close(self) -> None:
        """method for pipeline compliance"""
        pass

    def reset(self) -> None:
        """method for pipeline compliance"""
        pass
