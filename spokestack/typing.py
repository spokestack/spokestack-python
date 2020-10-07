"""
This module contains type aliases for the library
"""
from typing import Any, Dict, Union


Utterance = Union[str, None]
Intent = Union[str, None]
Slot = Union[Dict[Any, Any], None]
