"""
This module contains mappings to convert numbers in words
to integers. ie. five -> 5
"""

from typing import Dict


ENG_ZERO: Dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "owe": 0,
}
ENG_MOD10: Dict[str, int] = {
    "one": 1,
    "won": 1,
    "two": 2,
    "too": 2,
    "to": 2,
    "three": 3,
    "four": 4,
    "for": 4,
    "fore": 4,
    "five": 5,
    "six": 6,
    "sicks": 6,
    "sics": 6,
    "seven": 7,
    "eight": 8,
    "ate": 8,
    "nine": 9,
}
ENG_MOD20: Dict[str, int] = {
    "ten": 10,
    "tin": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

ENG_DIV10: Dict[str, int] = {
    "twenty": 2,
    "thirty": 3,
    "forty": 4,
    "fifty": 5,
    "sixty": 6,
    "seventy": 7,
    "eighty": 8,
    "ninety": 9,
}

ENG_EXP10: Dict[str, int] = {"hundred": 2, "thousand": 3}

MULTIPLIERS: Dict[str, int] = {
    "hundred": 100,
    "thousand": 1000,
    "million": int(1e6),
    "billion": int(1e9),
}
WORD_TO_NUM: Dict[str, int] = {
    **ENG_ZERO,
    **ENG_MOD10,
    **ENG_MOD20,
    **ENG_DIV10,
    **MULTIPLIERS,
    "first": 1,
    "second": 2,
    "third": 3,
    "fif": 5,
    "eigh": 8,
    "nin": 9,
    "twelf": 12,
    "twentie": 20,
    "thirtie": 30,
    "fortie": 40,
    "fiftie": 50,
    "seventie": 70,
    "eightie": 80,
    "ninetie": 90,
}
