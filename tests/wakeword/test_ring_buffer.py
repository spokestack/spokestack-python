"""
Tests for RingBuffer class
"""
import numpy as np
import pytest

from spokestack.ring_buffer import RingBuffer


def test_ring_buffer():
    buffer = RingBuffer([0, 1])
    assert buffer.capacity == 0
    assert buffer.is_empty
    assert buffer.is_full

    buffer = RingBuffer([1, 1])
    assert buffer.capacity == 1
    assert buffer.is_empty
    assert not buffer.is_full

    buffer = RingBuffer([10, 1])
    assert buffer.capacity == 10
    assert buffer.is_empty
    assert not buffer.is_full


def test_read_write():
    buffer = RingBuffer([3, 1])
    for i in range(buffer.capacity):
        buffer.write(np.ones((1, 1)) * (i + 1))

    assert not buffer.is_empty
    assert buffer.is_full

    # full write
    with pytest.raises(IndexError):
        buffer.write(np.ones((1, 1)))

    for i in range(buffer.capacity):
        assert buffer.read() == np.ones((1, 1)) * (i + 1)

    assert buffer.is_empty
    assert not buffer.is_full

    # empty read
    with pytest.raises(IndexError):
        buffer.read()


def test_rewind():
    buffer = RingBuffer([4, 1])

    # default
    buffer.rewind()
    assert not buffer.is_empty
    assert buffer.is_full
    while not buffer.is_empty:
        buffer.read()

    # valid
    for i in range(buffer.capacity):
        buffer.write(np.ones((1, 1)) * (i + 1))
    while not buffer.is_empty:
        buffer.read()

    buffer.rewind()
    assert not buffer.is_empty
    assert buffer.is_full

    for i in range(buffer.capacity):
        assert buffer.read() == np.ones((1, 1)) * (i + 1)

    assert buffer.is_empty
    assert not buffer.is_full


def test_seek():
    buffer = RingBuffer([5, 1])

    # valid
    for i in range(buffer.capacity):
        buffer.write(np.ones((1, 1)) * (i + 1))

    buffer.seek(1)
    for i in range(1, buffer.capacity):
        assert buffer.read() == np.ones((1, 1)) * (i + 1)
    buffer.rewind()

    buffer.seek(buffer.capacity - 1)
    for i in range((buffer.capacity - 1), buffer.capacity):
        assert buffer.read() == np.ones((1, 1)) * (i + 1)

    buffer.seek(-7)
    assert buffer.read() == np.ones((1, 1)) * buffer.capacity


def test_reset():
    buffer = RingBuffer([4, 1])
    buffer.fill(1)
    assert not buffer.is_empty
    assert buffer.is_full

    for i in range(buffer.capacity):
        assert buffer.read() == np.ones((1, 1), np.float32)

    buffer.reset()
    buffer.fill(1)
    assert not buffer.is_empty
    assert buffer.is_full

    for i in range(buffer.capacity - 1):
        assert buffer.read() == np.ones((1, 1))

    assert buffer.read() == np.ones((1, 1))


def test_read_all():
    buffer = RingBuffer([3, 1])
    values = []
    for i in range(buffer.capacity):
        inputs = np.ones((1, 1)) * (i + 1)
        buffer.write(inputs)
        values.append(inputs)

    assert not buffer.is_empty
    assert buffer.is_full

    all = buffer.read_all()
    assert buffer.is_empty
    assert not buffer.is_full
    assert all.all() == np.array(values).all()
