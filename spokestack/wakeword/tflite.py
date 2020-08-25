"""
This module contains the class for detecting
the presence of keywords in an audio stream
"""
import os

import numpy as np  # type: ignore
from numpy import hanning

from spokestack.context import SpeechContext
from spokestack.models.tensorflow import TFLiteModel
from spokestack.wakeword.ring_buffer import RingBuffer


class WakewordDetector:
    """ Detects the presence of a wakeword in the audio input

    Args:
            pre_emphasis (float): The value of the pre-emmphasis filter
            sample_rate (int): The number of audio samples per second of audio (kHz)
            fft_window_type (str): Window type: any that scipy.signal.get_window accepts
            fft_hop_length (int): Audio sliding window for STFT calculation (ms)
            model_dir (str): Path to the directory containing .tflite models
            posterior_threshold (float): Probability threshold for if a wakeword
                                         was detected
    """

    def __init__(
        self,
        pre_emphasis: float = 0.0,
        sample_rate: int = 16000,
        fft_window_type: str = "hann",
        fft_hop_length: int = 10,
        model_dir: str = "",
        posterior_threshold: float = 0.5,
    ) -> None:

        self.pre_emphasis: float = pre_emphasis
        self.sample_rate: int = sample_rate
        self.window_type: str = fft_window_type
        if self.window_type != "hann":
            raise ValueError("fft_window_type must be hann")

        self.hop_length: int = int(fft_hop_length * sample_rate / 1000)

        self.filter_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "filter.tflite")
        )
        self.encode_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "encode.tflite")
        )
        self.state = np.zeros(self.encode_model.input_details[1]["shape"], np.float32)
        self.detect_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "detect.tflite")
        )
        self.window_size: int = (
            self.filter_model.input_details[0]["shape"][-1] - 1
        ) * 2
        self._hann_window = hanning(self.window_size)
        self.mel_length: int = self.encode_model.input_details[0]["shape"][1]
        self.mel_width: int = self.encode_model.input_details[0]["shape"][-1]
        self.encode_length: int = self.detect_model.input_details[0]["shape"][1]
        self.encode_width: int = self.detect_model.input_details[0]["shape"][-1]

        self.sample_window: RingBuffer = RingBuffer(shape=[self.window_size])
        self.frame_window: RingBuffer = RingBuffer(
            shape=[self.mel_length, self.mel_width]
        )
        self.encode_window: RingBuffer = RingBuffer(
            shape=[self.encode_length, self.encode_width]
        )
        self.frame_window.fill(0.0)
        self.encode_window.fill(0.0)
        self._posterior_threshold: float = posterior_threshold
        self._posterior_max: float = 0.0
        self._prev_sample: float = 0.0
        self._is_speech: bool = False

    def __call__(self, context: SpeechContext, frame) -> None:
        """ Entry point of the detector

        Args:
            context (SpeechContext): current state of the speech pipeline
            frame (np.ndarray): a single frame of an audio signal

        Returns: None

        """
        vad_fall = self._is_speech and not context.is_speech
        self._is_speech = context.is_speech
        if not context.is_active:
            self._sample(context, frame)

        if vad_fall:
            self.reset()

    def _sample(self, context: SpeechContext, frame) -> None:

        frame = frame.astype(np.float32) / (2 ** 15 - 1)
        frame = np.clip(frame, -1.0, 1.0)
        prev_sample = frame[-1]
        frame -= self.pre_emphasis * np.append(self._prev_sample, frame[:-1])
        self._prev_sample = prev_sample

        for sample in frame:

            self.sample_window.write(sample)

            if self.sample_window.is_full:
                if context.is_speech:
                    self._analyze(context)
                self.sample_window.rewind().seek(self.hop_length)

    def _analyze(self, context: SpeechContext) -> None:
        frame = self.sample_window.read_all()
        frame = np.fft.rfft(frame * self._hann_window, n=self.window_size)
        frame = np.abs(frame).astype(np.float32)
        self._filter(context, frame)

    def _filter(self, context: SpeechContext, frame) -> None:
        frame = np.expand_dims(frame, 0)
        frame = self.filter_model(frame)[0]
        self.frame_window.rewind().seek(1)
        self.frame_window.write(frame)
        self._encode(context)

    def _encode(self, context: SpeechContext) -> None:
        frame = self.frame_window.read_all()
        frame = np.expand_dims(frame, 0)
        frame, self.state = self.encode_model(frame, self.state)

        self.encode_window.rewind().seek(1)
        self.encode_window.write(frame)

        self._detect(context)

    def _detect(self, context: SpeechContext) -> None:
        frame = self.encode_window.read_all()
        frame = np.expand_dims(frame, 0)
        posterior = self.detect_model(frame)[0]
        if posterior > self._posterior_threshold:
            context.is_active = True
        if posterior > self._posterior_max:
            self._posterior_max = posterior

    def reset(self) -> None:
        """ Resets the currect WakewordDetector state """
        self.sample_window.reset()
        self.frame_window.reset().fill(0.0)
        self.encode_window.reset().fill(0.0)
        self.state[:] = 0.0
        self._posterior_max = 0.0
