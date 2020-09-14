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

    raw_value = "oneth"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 1

    metadata = {"range": [1, 500]}
    raw_value = "four hundred four"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 404


def test_exponent():
    metadata = {"range": [1, 10000]}
    raw_value = "nine thousand one"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 9001

    raw_value = "one hundred"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 100


def test_teens():
    metadata = {"range": [1, 19]}
    raw_value = "thirteen"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 13

    raw_value = "13"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 13

    raw_value = "fourteen"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 14

    raw_value = "fifteen"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 15

    raw_value = "sixteen"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 16


def test_multiples_of_ten():
    metadata = {"range": [1, 100]}
    raw_value = "ten"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 10

    raw_value = "10"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 10

    raw_value = "20"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 20

    raw_value = "30"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 30

    raw_value = "44"
    parsed = integer.parse(metadata, raw_value)
    assert parsed == 44


def test_out_of_range():
    metadata = {"range": [1, 11]}
    raw_value = "twelve"
    parsed = integer.parse(metadata, raw_value)
    assert not parsed

    raw_value = "12"
    parsed = integer.parse(metadata, raw_value)
    assert not parsed

    raw_value = "100000000000"
    parsed = integer.parse(metadata, raw_value)
    assert not parsed


def test_invalid():
    metadata = {"range": [1, 400]}
    raw_value = "four hundred hundred"
    parsed = integer.parse(metadata, raw_value)
    assert not parsed

    raw_value = "word"
    parsed = integer.parse(metadata, raw_value)
    assert not parsed
