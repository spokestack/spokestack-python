"""
This module contains the tests for the selset parser
"""
from spokestack.nlu.parsers import selset


def test_selset():
    metadata = {
        "selections": [{"name": "lights", "aliases": ["lights", "beams", "bulbs"]}]
    }
    raw_value = "beams"
    parsed = selset.parse(metadata, raw_value)
    assert parsed == "lights"

    raw_value = "lights"
    parsed = selset.parse(metadata, raw_value)
    assert parsed == "lights"

    raw_value = "cat"
    parsed = selset.parse(metadata, raw_value)
    assert not parsed
