"""
This module contains the logic to parse digits from NLU results
"""
import re
from typing import Any, Dict, Union

from spokestack.nlu.parsers import maps


DIGIT_SPLIT_RE = re.compile("[-,()\\s]+")


def parse(metadata: Dict[str, Any], raw_value: str) -> str:
    """ Digit Parser

    Args:
        metadata (Dict[str, Any]): digit slot metadata
        raw_value (str): value tagged by the model

    Returns:
        (str): string parsed digits
    """
    count = metadata.get("count")
    normalized = raw_value.lower()
    tokens = DIGIT_SPLIT_RE.split(normalized)
    values = []
    for i, token in enumerate(tokens):
        next_token = None
        if i < len(tokens) - 1:
            next_token = tokens[i + 1]
        value = _parse_single(token, next_token)
        values.append(value)
    return "".join(values[:count])


def _parse_single(token: str, next_token: Union[str, None]) -> str:
    if token in maps.ENG_ZERO:
        return str(maps.ENG_ZERO[token])
    elif token in maps.ENG_MOD10:
        return str(maps.ENG_MOD10[token])
    elif token in maps.ENG_MOD20:
        return str(maps.ENG_MOD20[token])
    elif token in maps.ENG_DIV10 and next_token in maps.ENG_MOD10:
        return str(maps.ENG_DIV10[token])
    elif token in maps.ENG_DIV10:
        return str(maps.ENG_DIV10[token] * 10)
    elif token in maps.ENG_EXP10:
        exponent = maps.ENG_EXP10[token]
        return "".zfill(exponent)
    else:
        return ""
