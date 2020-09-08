"""
This module contains the logic to parse integers from NLU results
"""
from typing import Any, Dict, Union

from spokestack.nlu.parsers import maps
from spokestack.nlu.parsers.digits import DIGIT_SPLIT_RE


def parse(metadata: Dict[str, Any], raw_value: str) -> Union[int, None]:
    """ Parser for Integer slots

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
            reduced = _parse_reduce(token, parsed_values)
            if not reduced:
                return None

    result = sum(parsed_values)
    if _is_in_range(result, raw_range):
        return result
    return None


def _parse_reduce(number, so_far):
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


def _collapse(multiplier, so_far):
    collapsed = []
    total = 0
    for number in so_far:
        if number > multiplier:
            collapsed.append(number)
        else:
            total += number
    if not total > 0:
        total = 1
    collapsed.append(total * multiplier)
    return collapsed


def _is_in_range(value, interval):
    return value in range(interval[0], interval[1])
