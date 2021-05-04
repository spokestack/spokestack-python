"""
English NLP module for Spokestack TTS-Lite.
"""

import re
import typing as T

import inflect
import spacy
from unidecode import unidecode

_INFLECT = inflect.engine()

_ABBREVIATIONS = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "missus"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def nlp() -> spacy.Language:
    """ Create a spacy NLP object for this language. """

    return spacy.load("en_core_web_sm", disable=["ner", "textcat"])


def clean(text: str) -> str:
    """
    Preprocess a text utterance for TTS.

    Args:
        text (str): The utterance to preprocess

    Returns:
        str: The preprocessed text

    """
    text = _convert_to_ascii(text)
    text = _lowercase(text)
    text = _expand_numbers(text)
    text = _expand_abbreviations(text)
    text = _collapse_whitespace(text)
    return text


def _convert_to_ascii(text: str) -> str:
    return unidecode(text)


def _lowercase(text: str) -> str:
    return text.lower()


def _expand_numbers(text: str) -> str:
    text = re.sub(r"([0-9][0-9\,]+[0-9])", _remove_commas, text)
    text = re.sub(r"£([0-9\,]*[0-9]+)", r"\1 pounds", text)
    text = re.sub(r"\$([0-9\.\,]*[0-9]+)", _expand_dollars, text)
    text = re.sub(r"([0-9]+\.[0-9]+)", _expand_decimal_point, text)
    text = re.sub(r"[0-9]+(st|nd|rd|th)", _expand_ordinal, text)
    text = re.sub(r"[0-9]+", _expand_number, text)
    return text


def _expand_abbreviations(text: str) -> str:
    for regex, replacement in _ABBREVIATIONS:
        text = re.sub(regex, replacement, text)
    return text


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _remove_commas(match: T.Match) -> str:
    return match.group(1).replace(",", "")


def _expand_dollars(match: T.Match) -> str:
    group = match.group(1)
    parts = group.split(".")
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    if dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    if cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    return "zero dollars"


def _expand_decimal_point(match: T.Match) -> str:
    return match.group(1).replace(".", " point ")


def _expand_ordinal(match: T.Match) -> str:
    return _INFLECT.number_to_words(match.group(0))


def _expand_number(match: T.Match) -> str:
    num = int(match.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return "two thousand"
        if 2000 < num < 2010:
            return "two thousand " + _INFLECT.number_to_words(num % 100)
        if num % 100 == 0:
            return _INFLECT.number_to_words(num // 100) + " hundred"
        return _INFLECT.number_to_words(num, andword="", zero="oh", group=2).replace(
            ", ", " "
        )
    return _INFLECT.number_to_words(num, andword="")
