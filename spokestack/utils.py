"""
This module contains general utilities
"""
import json
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    """

    Args:
        path (str): path to json file

    Returns: dictionary of json content

    """
    with open(path) as file:
        config = json.load(file)
    return config
