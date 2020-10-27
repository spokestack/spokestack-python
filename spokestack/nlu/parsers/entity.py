"""
This module contains the logic to parse entities from NLU results. The entity parser
is a pass through for string values to allow custom logic to resolve the entities.
For example, the entity can be used as a keyword in a database search.
"""
from typing import Any, Dict


def parse(metadata: Dict[str, Any], raw_value: str) -> str:
    """Entity Parser

    Args:
        metadata (Dict[str, Any]): metadata for entity slot
        raw_value (str): tagged entity

    Returns:
        (str): tagged entity

    """
    return raw_value
