"""
This module contains the tests for the integer parser
"""
from spokestack.nlu.parsers import integer


def test_integer():
    metadata = {"range": [1, 50]}
    raw_value = "one"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 1

    raw_value = "1"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 1

    raw_value = "word"
    parsed = integer.parse(metadata, raw_value)
    assert not parsed

    raw_value = "oneth"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 1

    metadata = {"range": [1, 500]}
    raw_value = "four hundred four"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 404

    metadata = {"range": [1, 500]}
    raw_value = "four hundred hundred"
    parsed = integer.parse(metadata, raw_value)
    assert not parsed
