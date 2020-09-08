"""
This module contains the tests for the digits parser
"""
from spokestack.nlu.parsers import digits


def test_digits():
    metadata = {"count": 7}
    raw_value = "eight six seven five three oh nine"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "8675309"

    raw_value = "one eight hundred six seven eight"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "1800678"

    raw_value = "eight sixty seven five three oh nine"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "8675309"

    raw_value = "twenty three nineteen"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "2319"

    raw_value = "sixty five thousand one"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "650001"

    raw_value = "sixty"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "60"

    raw_value = ""
    parsed = digits.parse(metadata, raw_value)
    assert not parsed
