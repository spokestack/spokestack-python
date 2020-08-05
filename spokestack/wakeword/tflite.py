"""
This module contains the class for detecting
the presence of keywords in an audio stream
"""
import os

import numpy as np  # type: ignore
from scipy import signal  # type: ignore

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
        self.window_type: str = fft_window_type
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

        self.sample_window: RingBuffer = RingBuffer(shape=[self.window_size])
        self.fft_window: RingBuffer = RingBuffer(
            shape=[1, int(self.window_size / 2 + 1)]
        )
        self.frame_window: RingBuffer = RingBuffer(
            shape=[self.mel_length, self.mel_width]
        )
        self.encode_window: RingBuffer = RingBuffer(
            shape=[self.encode_length, self.encode_width]
        )
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
        self.prev_sample = 0.0

    def __call__(self, context: SpeechContext, frame) -> None:
        """ Entry point of the detector

        Args:
            context (SpeechContext): current state of the speech pipeline
            frame (np.ndarray): a single frame of an audio signal

        Returns: None

        """
        if not context.is_active:
            self.sample(context, frame)

        if not context.is_speech:
            self.reset()

    def sample(self, context: SpeechContext, frame) -> None:
        """ Iterates over samples from an audio frame

        Args:
            context (SpeechContext): current state of the speech pipeline
            frame (np.ndarray): a single frame of an audio signal

        Returns: None

        """
        for sample in frame:
            # normalize
            sample = sample / frame.max()
            # clip
            sample = np.clip(sample, -1.0, 1.0)
            # preemphasis
            prev_sample = frame[-1]
            sample -= self.pre_emphasis * np.hstack([[self.prev_sample], sample])[:-1]
            self.prev_sample = prev_sample

            self.sample_window.write(sample)
            if self.sample_window.is_full:
                if context.is_speech:
                    self._analyze(context)
                self.sample_window.rewind().seek(self.hop_length)

    def _analyze(self, context: SpeechContext) -> None:
        """ Applies preprocessing to a frame of audio

        Args:
            context (SpeechContext): current state of the speech pipeline

        Returns: None

        """

        frame = self.sample_window.to_array()
        window = signal.get_window("hann", self.window_size)
        frame = np.fft.rfft(frame * window, n=self.window_size)
        frame = np.abs(frame)
        self.sample_window.reset()

        self.fft_window.write(frame)
        self._filter(context)

    def _filter(self, context: SpeechContext) -> None:
        """ Converts a time domain signal into a melspectrogram

        Args:
            context (SpeechContext): current state of the speech pipeline

        Returns: None

        """

        self.fft_window.rewind()
        frame = self.fft_window.read()
        frame = self.filter_model([frame])[0]
        self.fft_window.reset()

        self.frame_window.rewind().seek(1)
        self.frame_window.write(frame)
        self._encode(context)

    def _encode(self, context: SpeechContext) -> None:
        """ Encodes each frame for input into the detector

        Args:
            context (SpeechContext): current state of the speech pipeline

        Returns: None

        """
        self.frame_window.rewind()
        frame = self.frame_window.to_array()
        frame = np.expand_dims(frame, 0)
        frame, self.state = self.encode_model([frame, self.state])

        self.encode_window.rewind().seek(1)
        self.encode_window.write(frame)

        self._detect(context)

    def _detect(self, context: SpeechContext) -> None:
        """ Detects the wakeword in a frame of audio

        Args:
            context (SpeechContext): current state of the speech pipeline

        Returns: None

        """

        frame = self.encode_window.to_array()
        frame = np.expand_dims(frame, 0)
        posterior = self.detect_model([frame])[0]
        if posterior > self._posterior_threshold:
            context.is_active = True
        if posterior > self._posterior_max:
            self._posterior_max = posterior

    def reset(self) -> None:
        """ Resets the currect WakewordDetector state """
        self.sample_window.reset()
        self.frame_window.reset().fill(0.0)
        self.encode_window.reset().fill(0.0)
        self.state = np.zeros(self.encode_model.input_details[1]["shape"], np.float32)
        self.fft_window.reset()
        self._posterior_max = 0.0

    def close(self) -> None:
        """ Closes the tflite models """
        self.filter_model.reset()
        self.encode_model.reset()
        self.detect_model.reset()
