"""
This module implements the RingBuffer class
"""
from typing import Any, Union

import numpy as np


class RingBuffer:
    """ ring buffer """

    def __init__(self, shape: list, dtype: Any = np.float32) -> None:
        self._dtype = dtype
        self._shape = shape
        self._shape[0] += 1
        self._buffer = np.empty(shape=self._shape, dtype=self._dtype)
        self._read: int = 0
        self._write: int = 0
        self._max_length = self._buffer.shape[0]

    @property
    def is_empty(self) -> bool:
        """Determines if buffer is empty

        Returns: True if empty, False otherwise

        """
        return self._read == self._write

    @property
    def is_full(self) -> bool:
        """Determines if the buffer is full

        Returns: True if full, False otherwise

        """
        return self._read == (self._write + 1) % self._max_length

    @property
    def capacity(self) -> int:
        """The capacity of the buffer

        Returns: Max size of the buffer

        """
        return self._max_length - 1

    def rewind(self) -> Any:
        """Rewinds the read head of the buffer to the most recent start position

        Returns: self

        """
        self._read = (self._write + 1) % self._max_length
        return self

    def reset(self) -> Any:
        """Empties the buffer

        Returns: self

        """
        self._write = self._read
        return self

    def fill(self, value: Union[int, float]) -> Any:
        """Fills the with a specific value

        Args:
            value (int or float): Fill value for the buffer

        Returns: self

        """
        self._buffer.fill(value)
        self._read = (self._write + 1) % self._max_length
        return self

    def seek(self, steps: int) -> Any:
        """Moves the read head a specified number of steps

        Args:
            steps (int): desired step length

        Returns: self

        """
        self._read = (self._read + steps) % self._max_length
        return self

    def write(self, item: np.ndarray) -> None:
        """Writes to the buffer and advances write head

        Args:
            item (np.ndarray): Array to be written to the buffer. Can be n-dimensional

        Returns: None

        """
        if self.is_full:
            raise IndexError("Buffer is full")

        self._buffer[self._write] = item
        self._write = (self._write + 1) % self._max_length

    def read(self) -> np.ndarray:
        """Reads from the buffer and advances read head.

        Returns: Array at the current read position

        """
        if self.is_empty:
            raise IndexError("Buffer is empty")

        item = self._buffer[self._read : self._read + 1]
        self._read = (self._read + 1) % self._max_length
        return item

    def read_all(self) -> np.ndarray:
        """Dumps the entire contents of the buffer to an array

        Returns: Array with full contents of the buffer

        """

        self.rewind()
        current = []
        while not self.is_empty:
            current.append(self.read())
        return np.concatenate(current).astype(self._dtype)
