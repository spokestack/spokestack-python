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

    metadata = {"count": 4}
    raw_value = "twenty three nineteen"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "2319"

    metadata = {"count": 6}
    raw_value = "sixty five thousand one"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "650001"

    metadata = {"count": 2}
    raw_value = "sixty"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "60"

    raw_value = ""
    parsed = digits.parse(metadata, raw_value)
    assert not parsed


def test_mod_twenty():
    metadata = {"count": 1}
    raw_values_single_digits = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]
    raw_values_double_digits = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]

    for i, raw_value in enumerate(raw_values_single_digits):
        parsed = digits.parse(metadata, raw_value)
        assert parsed == str(i + 1)

    metadata = {"count": 2}
    for i, raw_value in enumerate(raw_values_double_digits):
        parsed = digits.parse(metadata, raw_value)
        assert parsed == str(i + 10)


def test_div_ten():
    metadata = {"count": 2}
    raw_value = "twenty"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "20"

    raw_value = "forty one"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "41"

    raw_value = "fifty-two"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "52"

    raw_value = "sixty - eight"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "68"

    metadata = {"count": 4}
    raw_value = "thirty zero one"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "3001"

    raw_value = "seventy ten"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "7010"


def test_hundreds():
    metadata = {"count": 3}
    raw_value = "one hundred"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "100"

    metadata = {"count": 11}
    raw_value = "one eight hundred three three five twenty-two eleven"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "18003352211"

    raw_value = "18003352211"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "18003352211"


def test_thousands():
    metadata = {"count": 4}
    raw_value = "two thousand"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "2000"

    metadata = {"count": 11}
    raw_value = "one eight hundred three thirty-five four thousand"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "18003354000"


def test_homophones():
    metadata = {"count": 4}
    raw_value = "zero oh oh owe"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "0000"

    raw_value = "ten tin"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "1010"

    metadata = {"count": 2}
    raw_value = "one won"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "11"

    raw_value = "eight ate"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "88"

    metadata = {"count": 3}
    raw_value = "two too to"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "222"

    raw_value = "four for fore"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "444"

    raw_value = "six sicks sics"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "666"

    metadata = {"count": 12}
    raw_value = "oh one 23 for five 66 seventy-seven ate 9"
    parsed = digits.parse(metadata, raw_value)
    assert parsed == "012345667789"


def test_metadata_violation():
    metadata = {"count": 8}
    raw_value = "zero oh oh owe two two"
    parsed = digits.parse(metadata, raw_value)
    assert not parsed

    metadata = {"count": 1}
    parsed = digits.parse(metadata, raw_value)
    assert not parsed


def test_empty():
    metadata = {"count": 1}
    raw_value = "words"
    parsed = digits.parse(metadata, raw_value)
    assert not parsed
