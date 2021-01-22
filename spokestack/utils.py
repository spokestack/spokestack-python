"""
This module contains general utilities
"""
import json
from typing import Any, Dict

import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    """

    Args:
        path (str): path to json file

    Returns: dictionary of json content

    """
    with open(path) as file:
        config = json.load(file)
    return config


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    assert np.issubdtype(audio.dtype, np.int)
    return audio.astype(np.float32) / (2 ** 15 - 1)


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    assert np.issubdtype(audio.dtype, np.floating)
    return (audio * (2 ** 15 - 1)).astype(np.int16)
