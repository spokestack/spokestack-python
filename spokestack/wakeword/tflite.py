"""
This module contains the class for detecting
the presence of keywords in an audio stream
"""
import os

import numpy as np  # type: ignore

from spokestack.context import SpeechContext
from spokestack.models.tensorflow import TFLiteModel
from spokestack.wakeword.ring_buffer import RingBuffer


class WakewordTrigger:
    """ Detects the presence of a wakeword in the audio input

    Args:
            pre_emphasis (float): The value of the pre-emmphasis filter
            sample_rate (int): The number of audio samples per second of audio (kHz)
            fft_window_type (str): The type of fft window. (only support for hann)
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
        self.hop_length: int = int(fft_hop_length * sample_rate / 1000)

        if fft_window_type != "hann":
            raise ValueError("Invalid fft_window_type")

        self.filter_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "filter.tflite")
        )
        self.encode_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "encode.tflite")
        )
        # initialize the first state input for autoregressive encoder model
        self.state = np.zeros(self.encode_model.input_details[1]["shape"], np.float32)
        self.detect_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "detect.tflite")
        )

        # window size calculated based on fft
        # the filter inputs are (fft_size - 1) / 2
        # which makes the window size (post_fft_size - 1) * 2
        self._window_size = (self.filter_model.input_details[0]["shape"][-1] - 1) * 2
        self._fft_window = np.hanning(self._window_size)
        self.mel_length: int = self.encode_model.input_details[0]["shape"][1]
        self.mel_width: int = self.encode_model.input_details[0]["shape"][-1]
        self.encode_length: int = self.detect_model.input_details[0]["shape"][1]
        self.encode_width: int = self.detect_model.input_details[0]["shape"][-1]

        self.sample_window: RingBuffer = RingBuffer(shape=[self._window_size])
        self.frame_window: RingBuffer = RingBuffer(
            shape=[self.mel_length, self.mel_width]
        )
        self.encode_window: RingBuffer = RingBuffer(
            shape=[self.encode_length, self.encode_width]
        )
        # initialize frame window with zeros
        self.frame_window.fill(0.0)
        # initialize encoder window with zeros
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
        # detect vad edges for wakeword deactivation
        vad_fall = self._is_speech and not context.is_speech
        self._is_speech = context.is_speech

        # sample frame to detect the presence of wakeword
        if not context.is_active:
            self._sample(context, frame)

        # reset on vad fall deactivation
        if vad_fall:
            self.reset()

    def _sample(self, context: SpeechContext, frame) -> None:
        # normalize incoming audio to (-1.0, 1.0)
        frame = frame.astype(np.float32) / (2 ** 15 - 1)
        # ensure range does not exceed (-1.0, 1.0)
        frame = np.clip(frame, -1.0, 1.0)
        # pull out a single value from the frame
        prev_sample = frame[-1]
        # apply pre-emphasis with the previous sample
        frame -= self.pre_emphasis * np.append(self._prev_sample, frame[:-1])
        # cache the previous sample to be use in the next iteration
        self._prev_sample = prev_sample

        for sample in frame:
            self.sample_window.write(sample)
            # fill the sample window
            if self.sample_window.is_full:
                # run analyze if speech detected in the vad
                if context.is_speech:
                    self._analyze(context)
                # prepare the sample window for the next iteration
                self.sample_window.rewind().seek(self.hop_length)

    def _analyze(self, context: SpeechContext) -> None:
        # read the full contents of the sample window
        frame = self.sample_window.read_all()
        # apply real valued fft
        frame = np.fft.rfft(frame * self._fft_window, n=self._window_size)
        # polarize and cast from float64 to float32
        frame = np.abs(frame).astype(np.float32)
        # compute mel spectrogram
        self._filter(context, frame)

    def _filter(self, context: SpeechContext, frame) -> None:
        # add the batch dimension
        frame = np.expand_dims(frame, 0)
        # compute the mel spectrogram with filter model
        frame = self.filter_model(frame)[0]
        # write to the frame window
        self.frame_window.rewind().seek(1)
        self.frame_window.write(frame)
        # encode the mel spectrogram
        self._encode(context)

    def _encode(self, context: SpeechContext) -> None:
        # read the full contents of the frame window
        frame = self.frame_window.read_all()
        # add batch dimension
        frame = np.expand_dims(frame, 0)
        # run the encoder and save the output state for autoregression
        frame, self.state = self.encode_model(frame, self.state)
        # accumulate encoded samples until size of detection window
        self.encode_window.rewind().seek(1)
        self.encode_window.write(frame)
        self._detect(context)

    def _detect(self, context: SpeechContext) -> None:
        # read the full contents of the encode window
        frame = self.encode_window.read_all()
        # add the batch dimension
        frame = np.expand_dims(frame, 0)
        # get scalar probability of wakeword
        posterior = self.detect_model(frame)[0][0][0]
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
