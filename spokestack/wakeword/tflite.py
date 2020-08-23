"""
This module contains the class for detecting
the presence of keywords in an audio stream
"""
import os

import numpy as np  # type: ignore

from spokestack.context import SpeechContext
from spokestack.models.tensorflow import TFLiteModel
from spokestack.wakeword.ring_buffer import RingBuffer


class WakewordDetector:
    """ Detects the presence of a wakeword in the audio input

    Args:
            pre_emphasis (float): The value of the pre-emmphasis filter
            sample_rate (int): The number of audio samples per second of audio (kHz)
            fft_window_size (int): Size of the sliding window for STFT calculation
            fft_window_type (str): Window type: any that scipy.signal.get_window accepts
            fft_hop_length (int): Audio sliding window for STFT calculation (ms)
            mel_frame_length (int): Frame length of the mel spectrogram (samples)
            mel_frame_width (int): The number of features in the mel spectrogram
            wake_encode_length (int): The length of the encoder output sliding
                                      window (ms)
            wake_encode_width (int): The number of features in each encoded frame
            model_dir (str): Path to the directory containing .tflite models
            posterior_threshold (float): Probability threshold for if a wakeword
                                         was detected
    """

    def __init__(
        self,
        pre_emphasis: float = 0.0,
        sample_rate: int = 16000,
        fft_window_size: int = 512,
        fft_window_type: str = "hann",
        fft_hop_length: int = 10,
        mel_frame_length: int = 10,
        mel_frame_width: int = 40,
        wake_encode_length: int = 1000,
        wake_encode_width: int = 128,
        model_dir: str = "",
        posterior_threshold: float = 0.5,
    ) -> None:

        self.pre_emphasis: float = pre_emphasis
        self.sample_rate: int = sample_rate
        self.window_size: int = fft_window_size

        if self.window_size % 2 != 0:
            raise ValueError("fft_window_size must be divisible by 2")

        self.window_type: str = fft_window_type
        if self.window_type != "hann":
            raise ValueError("fft_window_type must be hann")

        self.hop_length: int = int(fft_hop_length * sample_rate / 1000)
        self.mel_length: int = int(
            mel_frame_length * sample_rate / 1000 / self.hop_length
        )
        self.mel_width: int = mel_frame_width
        self.encode_length: int = int(
            wake_encode_length * sample_rate / 1000 / self.hop_length
        )
        self.encode_width: int = wake_encode_width
        self.state_width: int = wake_encode_width
        self._hann_window = np.hanning(self.window_size)

        self.sample_window: RingBuffer = RingBuffer(shape=[self.window_size])
        self.fft_frame = np.empty((1, int(self.window_size / 2 + 1)), np.float32)
        self.frame_window: RingBuffer = RingBuffer(
            shape=[self.mel_length, self.mel_width]
        )
        self.encode_window: RingBuffer = RingBuffer(
            shape=[self.encode_length, self.encode_width]
        )
        self.frame_window.fill(np.zeros((1, self.mel_width), np.float32))
        self.encode_window.fill(np.zeros((1, self.encode_width), np.float32))
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

        frame /= 2 ** 15 - 1
        frame = np.clip(frame, -1.0, 1.0)
        frame = np.append(frame[0], frame[1:] - self.pre_emphasis * frame[:-1])

        for sample in frame:

            self.sample_window.write(sample)

            if self.sample_window.is_full:
                if context.is_speech:
                    self._analyze(context)
                self.sample_window.rewind().seek(self.hop_length)

    def _analyze(self, context: SpeechContext) -> None:
        frame = self.sample_window.read_all()
        frame = np.fft.rfft(frame * self._hann_window, n=self.window_size)
        frame = np.abs(frame)

        self.fft_frame[0] = frame

        self._filter(context)

    def _filter(self, context: SpeechContext) -> None:
        frame = self.filter_model(self.fft_frame)[0]
        self.frame_window.rewind().seek(1)
        self.frame_window.write(frame)
        self._encode(context)

    def _encode(self, context: SpeechContext) -> None:
        frame = self.frame_window.read_all()
        frame = np.expand_dims(frame, 0)
        frame, self.state = self.encode_model([frame, self.state])

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
        self.frame_window.reset().fill(np.zeros((1, self.mel_width), np.float32))
        self.encode_window.reset().fill(np.zeros((1, self.encode_width), np.float32))
        self.state[:] = 0.0
        self.fft_frame[:] = 0.0
        self._posterior_max = 0.0
