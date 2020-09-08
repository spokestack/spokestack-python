"""
This module contains the logic to parse entities from NLU results
"""
from typing import Any, Dict


def parse(metadata: Dict[str, Any], raw_value: str) -> str:
    """ Entity Parser

    Args:
        metadata (Dict[str, Any]): metadata for entity slot
        raw_value (str): tagged entity

    Returns:
        (str): tagged entity

    """
    return raw_value
