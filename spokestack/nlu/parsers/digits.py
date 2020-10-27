"""
This module contains the parser that converts the string representation
of a sequence of digits into the corresponding sequence of digits. These digits may
be in the form of english cardinal representations of numbers, along with some
homophones. The digits can be hyphenated or unhyphenated from twenty through
ninety-nine. The unhyphenated numbers get joined automatically. The use of
unhyphenated numbers introduces ambiguity. For example, "sixty five thousand" could
be parsed as "605000" or "65000". Our parser will output the latter. However, this
can be an issue with values such as "sixty five thousand one" which parses as "650001".
This limitation will most likely be acceptable for most multi-digit use cases such as
telephone numbers, social security numbers, etc.
"""
from typing import Any, Dict, Union

from spokestack.nlu.parsers import DIGIT_SPLIT_RE, maps


def parse(metadata: Dict[str, Any], raw_value: str) -> str:
    """Digit Parser

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

    combined = "".join(values)
    if count:
        if len(combined) != count:
            return ""
    return combined


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
        return "0" * exponent
    else:
        try:
            return str(int(token))
        except ValueError:
            return ""
