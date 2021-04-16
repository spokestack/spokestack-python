"""Tests for sounddevice based input."""
from unittest import mock

import numpy as np

from spokestack.io import sound_device


@mock.patch("spokestack.io.sound_device.sd")
def test_input(*args):
    mic = sound_device.SoundDeviceInput(sample_rate=16000, frame_width=10)
    mic._stream.read.return_value = np.zeros(160, np.int16).tobytes()
    # start audio stream
    mic.start()
    assert mic.is_active
    # read a single frame
    _ = mic.read()
    mic._stream.read.assert_called()
    # stop audio stream
    mic.stop()
    assert mic.is_stopped
    mic.close()
