"""
This module contains the class for detecting
the presence of keywords in an audio stream
"""
import logging
import os
from typing import Any

import numpy as np

from spokestack.context import SpeechContext
from spokestack.models.tensorflow import TFLiteModel
from spokestack.ring_buffer import RingBuffer

_LOG = logging.getLogger(__name__)


class WakewordTrigger:
    """Detects the presence of a wakeword in the audio input

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
        **kwargs: Any,
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
        self.detect_model: TFLiteModel = TFLiteModel(
            model_path=os.path.join(model_dir, "detect.tflite")
        )

        # window size calculated based on fft
        # the filter inputs are (fft_size - 1) / 2
        # which makes the window size (post_fft_size - 1) * 2
        self._window_size = (self.filter_model.input_details[0]["shape"][-1] - 1) * 2
        self._fft_window = np.hanning(self._window_size)

        # retrieve the mel_length and mel_width based on the encoder model metadata
        # these allocate the buffer to the correct size
        self.mel_length: int = self.encode_model.input_details[0]["shape"][1]
        self.mel_width: int = self.encode_model.input_details[0]["shape"][-1]

        # initialize the first state input for autoregressive encoder model
        # retrieve the encode_length and encode_width from the model detect_model
        # metadata. We get the dimensions from the detect_model inputs because the
        # encode_model runs autoregressively and outputs a single encoded sample.
        # the detect_model input is a collection of these samples.
        self.state = np.zeros(self.encode_model.input_details[1]["shape"], np.float32)
        self.encode_length: int = self.detect_model.input_details[0]["shape"][1]
        self.encode_width: int = self.detect_model.input_details[0]["shape"][-1]

        self.sample_window: RingBuffer = RingBuffer(shape=[self._window_size])
        self.frame_window: RingBuffer = RingBuffer(
            shape=[self.mel_length, self.mel_width]
        )
        self.encode_window: RingBuffer = RingBuffer(
            shape=[self.encode_length, self.encode_width]
        )

        # initialize the frame and encode windows with zeros
        # this minimizes the delay caused by filling the buffer
        self.frame_window.fill(0.0)
        self.encode_window.fill(-1.0)

        self._posterior_threshold: float = posterior_threshold
        self._posterior_max: float = 0.0
        self._prev_sample: float = 0.0
        self._is_speech: bool = False

    def __call__(self, context: SpeechContext, frame: np.ndarray) -> None:
        """Entry point of the trigger

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
            if not context.is_active:
                _LOG.info(f"wake: {self._posterior_max}")
            self.reset()

    def _sample(self, context: SpeechContext, frame: np.ndarray) -> None:
        # convert the PCM-16 audio to float32 in (-1.0, 1.0)
        frame = frame.astype(np.float32) / (2 ** 15 - 1)
        frame = np.clip(frame, -1.0, 1.0)

        # pull out a single value from the frame and apply pre-emphasis
        # with the previous sample then cache the previous sample
        # to be use in the next iteration
        prev_sample = frame[-1]
        frame -= self.pre_emphasis * np.append(self._prev_sample, frame[:-1])
        self._prev_sample = prev_sample

        # fill the sample window to analyze speech containing samples
        # after each window fill the buffer advances by the hop length
        # to produce an overlapping window
        for sample in frame:
            self.sample_window.write(sample)
            if self.sample_window.is_full:
                if context.is_speech:
                    self._analyze(context)
                self.sample_window.rewind().seek(self.hop_length)

    def _analyze(self, context: SpeechContext) -> None:
        # read the full contents of the sample window to calculate a single frame
        # of the STFT by applying the DFT to a real-valued input and
        # taking the magnitude of the complex DFT
        frame = self.sample_window.read_all()
        frame = np.fft.rfft(frame * self._fft_window, n=self._window_size)
        frame = np.abs(frame).astype(np.float32)

        # compute mel spectrogram
        self._filter(context, frame)

    def _filter(self, context: SpeechContext, frame: np.ndarray) -> None:
        # add the batch dimension and compute the mel spectrogram with filter model
        frame = np.expand_dims(frame, 0)
        frame = self.filter_model(frame)[0]

        # advance the window by 1 and write mel frame to the frame buffer
        self.frame_window.rewind().seek(1)
        self.frame_window.write(frame)

        # encode the mel spectrogram
        self._encode(context)

    def _encode(self, context: SpeechContext) -> None:
        # read the full contents of the frame window and add the batch dimension
        # run the encoder and save the output state for autoregression
        frame = self.frame_window.read_all()
        frame = np.expand_dims(frame, 0)
        frame, self.state = self.encode_model(frame, self.state)

        # accumulate encoded samples until size of detection window
        self.encode_window.rewind().seek(1)
        self.encode_window.write(frame)
        self._detect(context)

    def _detect(self, context: SpeechContext) -> None:
        # read the full contents of the encode window and add the batch dimension
        # calculate a scalar probability of if the frame contains the wakeword
        # with the detect model
        frame = self.encode_window.read_all()
        frame = np.expand_dims(frame, 0)
        posterior = self.detect_model(frame)[0][0][0]

        if posterior > self._posterior_max:
            self._posterior_max = posterior
        if posterior > self._posterior_threshold:
            context.is_active = True
            _LOG.info(f"wake: {self._posterior_max}")

    def reset(self) -> None:
        """ Resets the currect WakewordDetector state """
        self.sample_window.reset()
        self.frame_window.reset().fill(0.0)
        self.encode_window.reset().fill(-1.0)
        self.state[:] = 0.0
        self._posterior_max = 0.0

    def close(self) -> None:
        """ Close interface for use in the pipeline """
        self.reset()
