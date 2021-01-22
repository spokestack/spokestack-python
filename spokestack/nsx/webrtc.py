"""
This module contains the class for webrtc automatic noise suppression
"""
import numpy as np

from spokestack.context import SpeechContext
from spokestack.extensions.webrtc.nsx import WebRtcNsx

POLICY_MILD = 0
POLICY_MEDIUM = 1
POLICY_AGGRESSIVE = 2
POLICY_VERY_AGGRESSIVE = 3


class AutomaticNoiseSuppression:
    """WebRTC Automatic Noise Suppression

    Args:
        sample_rate (int): audio sample rate. (Hz)
        policy (int): aggressiveness of the noise suppression:
                      - POLICY_MILD: mild suppression (6dB)
                      - POLICY_MEDIUM: medium suppression (10dB)
                      - POLICY_AGGRESSIVE: aggresive suppression (15dB)
                      - POLICY_VERY_AGGRESSIVE: very aggressive suppression
    """

    def __init__(
        self, sample_rate: int = 16000, policy: int = POLICY_MEDIUM, **kwargs: str
    ) -> None:

        # validate sample rate
        if sample_rate not in {8000, 16000, 32000}:
            raise ValueError("sample_rate")

        self._nsx = WebRtcNsx(sample_rate=sample_rate, policy=policy)
        self._frame_width = sample_rate * 10 // 1000

    def __call__(self, context: SpeechContext, frame: np.array) -> None:
        """Main Entry Point

        Args:
            context (SpeechContext): State based information that needs to be shared
            between pieces of the pipeline
            frame (np.array): PCM-16 Audio
        """
        frame_size = self._frame_width
        # validate frame size
        if len(frame) % frame_size != 0:
            raise ValueError("invalid_frame_size")
        # validate dtype
        if not np.issubdtype(frame.dtype, np.signedinteger):
            raise TypeError("invalid_dtype")

        for i in range(len(frame) // frame_size):
            self._nsx(frame[i : frame_size + i])

    def close(self) -> None:
        """method for pipeline compliance"""
        pass

    def reset(self) -> None:
        """method for pipeline compliance"""
        pass
