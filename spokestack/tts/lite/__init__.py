"""
Spokestack-Lite Speech Synthesizer

This module contains the SpeechSynthesizer class used to convert text to speech
using local TTS models trained on the Spokestack platform. A SpeechSynthesizer
instance can be passed to the TextToSpeechManager for playback.

Example:
    This example assumes that a TTS model was downloaded from the Spokestack
    platform and extracted to the :code:`model` directory. ::

        from spokestack.io.pyaudio import PyAudioOutput
        from spokestack.tts.manager import TextToSpeechManager, FORMAT_PCM16
        from spokestack.tts.lite import SpeechSynthesizer, BLOCK_LENGTH, SAMPLE_RATE

        tts = TextToSpeechManager(
            SpeechSynthesizer("./model"),
            PyAudioOutput(sample_rate=SAMPLE_RATE, frames_per_buffer=BLOCK_LENGTH),
            format_=FORMAT_PCM16)

        tts.synthesize("Hello world!")

"""

import importlib
import json
import os
import re
import typing as T
from collections import defaultdict

import numpy as np

from spokestack.models.tensorflow import TFLiteModel

# signal configuration
SAMPLE_RATE = 24000
HOP_LENGTH = 240
ENCODER_PAD = -2
BREAK_LENGTH = 0.1

# streaming/cross-fading configuration
FRAME_LENGTH = 63
FRAME_OVERLAP = 1
BLOCK_LENGTH = FRAME_LENGTH * HOP_LENGTH
BLOCK_OVERLAP = FRAME_OVERLAP * HOP_LENGTH
FADE_OUT = np.linspace(1, 0, BLOCK_OVERLAP, dtype=np.float32)
FADE_IN = FADE_OUT[::-1]


class SpeechSynthesizer:
    """
    Initialize a new lightweight speech synthesizer

    Args:
        model_path (str): Path to the extracted TTS model downloaded from the
            Spokestack platform

    """

    def __init__(self, model_path: str):
        # load NLP configuration
        self._lexicon = _load_lexicon(os.path.join(model_path, "lexicon.txt"))

        with open(os.path.join(model_path, "metadata.json")) as file:
            metadata = json.load(file)

        lang = metadata["language"]
        self._sym_to_id = {s: i for i, s in enumerate(metadata["alphabet"])}
        self._language: T.Any = importlib.import_module(f"spokestack.tts.lite.{lang}")
        self._nlp = self._language.nlp()

        # load the TTS models
        self._aligner = TFLiteModel(os.path.join(model_path, "align.tflite"))
        self._encoder = TFLiteModel(os.path.join(model_path, "encode.tflite"))
        self._decoder = TFLiteModel(os.path.join(model_path, "decode.tflite"))
        self._aligner_input_index = self._aligner.input_details[0]["index"]
        self._encoder_input_index = self._encoder.input_details[0]["index"]

    def synthesize(
        self, utterance: str, *_args: T.List, **_kwargs: T.Dict
    ) -> T.Iterator[np.array]:
        """
        Synthesize a text utterance to speech audio

        Args:
            utterance (str): The text string to synthesize

        Returns:
            Iterator[np.array]: A generator for returns a sequence of
            PCM-16 numpy audio blocks for playback, storage, etc.

        """

        # segment sentences into a list of phoneme/grapheme lists
        for tokens in self._parse(utterance):
            # convert tokens to a vector of ids
            inputs = self._vectorize(tokens)

            # run the aligner model
            self._aligner.resize(self._aligner_input_index, inputs.shape)
            inputs = self._aligner(inputs)[0]

            # run the encoder model
            self._encoder.resize(self._encoder_input_index, inputs.shape)
            encoded = self._encoder(inputs)[0]

            # stream the decoder model and cross-fade the output audio
            overlap = np.zeros([BLOCK_OVERLAP], dtype=np.float32)
            for i in range(FRAME_OVERLAP, len(encoded), FRAME_LENGTH):
                # decode the current frame, padding as need to fill the decoder's input
                inputs = encoded[i - FRAME_OVERLAP : i + FRAME_LENGTH]
                inputs = np.pad(
                    inputs,
                    [(0, (FRAME_LENGTH + FRAME_OVERLAP) - len(inputs)), (0, 0)],
                    "constant",
                    constant_values=ENCODER_PAD,
                )
                outputs = self._decoder(inputs)[0]

                # fade in the new block, convert to int16 and return it
                overlap += outputs[:BLOCK_OVERLAP] * FADE_IN
                block = np.hstack([overlap, outputs[BLOCK_OVERLAP:-BLOCK_OVERLAP]])
                yield (block * (2 ** 15 - 1)).astype(np.int16)

                # fade out the previous block for mixing with the next block
                overlap = outputs[-BLOCK_OVERLAP:] * FADE_OUT

            # add a break after each segment
            yield np.zeros([int(BREAK_LENGTH * SAMPLE_RATE)], dtype=np.int16)

    def _parse(self, text: str) -> T.Iterator[str]:
        # perform language-specific number conversions, abbreviation expansions, etc.
        text = self._language.clean(text)

        # escape characters used for phonetic substitution
        text = re.sub(r"{", "[", text)
        text = re.sub(r"}", "]", text)

        # segment and tokenize the text, and convert words to their phonetic
        # representations using the attached lexicon
        for sentence in self._nlp(text).sents:
            tokens = []
            for token in sentence:
                if token.pos_ in ["SYM", "PUNCT"]:
                    tokens.append(token.text_with_ws)
                else:
                    entry = self._lexicon.get(token.text.lower(), {})
                    ipa = entry.get(token.tag_, entry.get(None))
                    tokens.append(
                        f"{{{ipa}}}{token.whitespace_}" if ipa else token.text_with_ws
                    )
            yield re.sub(r"}\s+{", " ", "".join(tokens))

    def _vectorize(self, text: str) -> np.array:
        # start with bos token
        vector = [self._sym_to_id["^"]]

        while text:
            # check for curly braces and treat their contents as ipa
            matches = re.match(r"(.*?)\{(.+?)\}(.*)", text)

            # no ipa in this block, vectorize graphemes
            if not matches:
                vector.extend(self._vectorize_text(text))
                break

            # ipa found, vectorize leading text, then phones
            vector.extend(self._vectorize_text(matches.group(1)))
            vector.extend(self._vectorize_phones(matches.group(2)))
            text = matches.group(3)

        # append eos token
        vector.append(self._sym_to_id["~"])
        return np.array(vector, dtype=np.int32)

    def _vectorize_text(self, text: T.Union[str, T.List[str]]) -> T.List[int]:
        return [
            self._sym_to_id[c] for c in text if c in self._sym_to_id and c not in "_^~"
        ]

    def _vectorize_phones(self, phones: str) -> T.List[int]:
        return self._vectorize_text([f"@{c}" if c != " " else c for c in phones])


def _load_lexicon(path: str) -> T.Dict[str, T.Dict[T.Optional[str], str]]:
    lexicon: T.Dict[str, T.Dict[T.Optional[str], str]] = defaultdict(dict)

    with open(path, "r") as file:
        for line in file:
            # parse the the lexicon entry, discard any alternative pronunciations
            parts = line.strip().split("\t")
            if len(parts) > 1:
                word = parts[0].lower()
                ipa = parts[1].split(",")[0].strip()
                pos = parts[2] if len(parts) > 2 else None
                lexicon[word][pos] = ipa

    return lexicon
