"""
This module contains the tests for the entity parser
"""
from spokestack.nlu.parsers import entity


def test_entity_real():
    metadata = {}
    raw_value = "im a real entity"
    parsed = entity.parse(metadata, raw_value)
    assert parsed == raw_value


def test_entity_empty():
    metadata = {}
    raw_value = ""
    parsed = entity.parse(metadata, raw_value)
    assert parsed == raw_value
