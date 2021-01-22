"""
This module contains general purpose utilities for webrtc extensions
"""


class ProcessError(Exception):
    """Error for failure in processing"""

    def __init__(self, message: str):
        super().__init__(message)
