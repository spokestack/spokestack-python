"""
This module contains the logic to parse selsets from NLU results. Selsets contain a name
along with one or more aliases. This allows one to map any of the listed aliases
into a single word. For example, if a selset's name is "light", and its aliases are
bulbs, light, beam, lamp, etc., occurrences of any alias will be parsed as light
"""
from typing import Any, Dict, Union


def parse(metadata: Dict[str, Any], raw_value: str) -> Union[str, None]:
    """Selset Parser

    Args:
        metadata (Dict[str, Any]): slot metadata
        raw_value (str): value tagged by the model

    Returns:
        Union[str, None]: selset or None if invalid
    """
    normalized = raw_value.lower()
    selections = metadata.get("selections", [])
    for selection in selections:
        name = selection.get("name")
        if name.lower() == normalized:
            return name
        aliases = selection.get("aliases")
        for alias in aliases:
            if alias.lower() == normalized:
                return name
    return None
