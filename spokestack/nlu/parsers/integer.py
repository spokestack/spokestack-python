"""
This module contains the logic to parse integers from NLU results. Integers can be
in the form of words (ie. one, two, three) or numbers (ie. 1, 2, 3). Either form
will resolve to Python's built-in 'int' type. The metadata must contain a range
key containing the minimum and maximum values for the expected integer range. It is
important to note the difference between digits and integers. Integers are
counting numbers: 2 apples, a table for two. In contrast, digits
can be used for sequences of numbers like phone numbers or social security numbers.
"""
from typing import Any, Dict, Union

from spokestack.nlu.parsers import DIGIT_SPLIT_RE, maps


def parse(metadata: Dict[str, Any], raw_value: str) -> Union[int, None]:
    """Integer Parser

    Args:
        metadata (Dict[str, Any]): metadata for the integer slot
        raw_value (str): value tagged by the model

    Returns:
        Union[int, None]: integer if parsable, None if invalid
    """
    raw_range = metadata.get("range")
    normalized = raw_value.lower()
    tokens = DIGIT_SPLIT_RE.split(normalized)

    parsed_values = []
    for token in tokens:
        try:
            parsed = int(token)
            parsed_values.append(parsed)
        except ValueError:
            if not _parse_reduce(token, parsed_values):
                return None

    result = sum(parsed_values)
    if _is_in_range(result, raw_range):
        return result
    return None


def _parse_reduce(number: Any, so_far: Any) -> Any:
    to_parse = number
    if to_parse.endswith("th"):
        to_parse = to_parse[: len(to_parse) - 2]
    if to_parse not in maps.WORD_TO_NUM:
        return None
    if to_parse in maps.MULTIPLIERS:
        total = _collapse(maps.MULTIPLIERS[to_parse], so_far)
        so_far.clear()
        so_far += total
    else:
        so_far.append(maps.WORD_TO_NUM[to_parse])
    return so_far


def _collapse(multiplier: int, so_far: Any) -> Any:
    collapsed = []
    total = 0
    for number in so_far:
        if number > multiplier:
            collapsed.append(number)
        else:
            total += number
    total = max(total, 1)
    collapsed.append(total * multiplier)
    return collapsed


def _is_in_range(value: int, interval: Any) -> bool:
    return value in range(interval[0], interval[1])
