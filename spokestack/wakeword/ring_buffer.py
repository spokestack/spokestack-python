"""
This module implements the RingBuffer class
"""

import numpy as np  # type: ignore


class RingBuffer:
    """ ring buffer """

    def __init__(self, shape: list, dtype=np.float32) -> None:
        shape[0] += 1
        self._buffer = np.empty(shape=shape, dtype=dtype)
        self._read: int = 0
        self._write: int = 0
        self._max_length = self._buffer.shape[0]

    @property
    def is_empty(self) -> bool:
        """ Determines if buffer is empty

        Returns: True if empty, False otherwise

        """
        return self._read == self._write

    @property
    def is_full(self) -> bool:
        """ Determines if the buffer is full

        Returns: True if full, False otherwise

        """
        return self._read == (self._write + 1) % self._max_length

    @property
    def capacity(self) -> int:
        """ The capacity of the buffer

        Returns: Max size of the buffer

        """
        return self._max_length - 1

    def rewind(self):
        """ Rewinds the read head of the buffer to the most recent start position

        Returns: self

        """
        self._read = (self._write + 1) % self._max_length
        return self

    def reset(self):
        """ Empties the buffer

        Returns: self

        """
        self._write = self._read
        assert self.is_empty
        return self

    def fill(self, value: np.ndarray):
        """ Fills the with a specific value

        Args:
            value (np.ndarray): Fill value for the buffer

        Returns: self

        """
        while not self.is_full:
            self.write(value)
        return self

    def seek(self, steps: int):
        """ Moves the read head a specified number of steps

        Args:
            steps (int): desired step length

        Returns: self

        """
        self._read = (self._read + steps) % self._max_length
        return self

    def write(self, item: np.ndarray) -> None:
        """ Writes to the buffer and advances write head

        Args:
            item (np.ndarray): Array to be written to the buffer. Can be n-dimensional

        Returns: None

        """
        if self.is_full:
            raise OverflowError("Buffer is full")

        self._buffer[self._write] = item
        self._write = (self._write + 1) % self._max_length

    def read(self) -> np.ndarray:
        """ Reads from the buffer and advances read head.

        Returns: Array at the current read position

        """
        if self.is_empty:
            raise IndexError("Buffer is empty")

        # pull item from current tail index
        item = self._buffer[self._read : self._read + 1]
        # calc new tail index
        self._read = (self._read + 1) % self._max_length
        return item

    def to_array(self) -> np.ndarray:
        """ Dumps the entire contents of the buffer to an array

        Returns: Array with full contents of the buffer

        """
        return self._buffer[: self.capacity]
