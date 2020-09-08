"""
This module contains the class to serve TFLite NLU models
"""
import json
import os
from importlib import import_module
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore
from tokenizers import BertWordPieceTokenizer  # type: ignore

from spokestack import utils
from spokestack.models.tensorflow import TFLiteModel


class TFLiteNLU:
    """ Abstraction for using TFLite NLU models

    Args:
        model_dir (str): path to the model directory containing nlu.tflite,
                         metadata.json, and vocab.txt
    """

    def __init__(self, model_dir: str) -> None:
        self._model = TFLiteModel(model_path=os.path.join(model_dir, "nlu.tflite"))
        self._metadata = utils.load_json(os.path.join(model_dir, "metadata.json"))
        self._tokenizer = BertWordPieceTokenizer(os.path.join(model_dir, "vocab.txt"))
        self._max_length = self._model.input_details[0]["shape"][-1]
        self._intent_decoder = {
            i: intent["name"] for i, intent in enumerate(self._metadata["intents"])
        }
        self._tag_decoder = {i: tag for i, tag in enumerate(self._metadata["tags"])}
        self._intent_meta = {
            intent.pop("name"): intent for intent in self._metadata["intents"]
        }
        self._warm_up()

    def __call__(self, utterance: str) -> Dict[str, Any]:
        """ Forward Pass

        Args:
            utterance (str): string that needs to be understood

        Returns: intents, slots, and model confidence

        """
        inputs, input_ids = self._encode(utterance)
        outputs = self._model(inputs)
        intent, tags, confidence = self._decode(outputs)

        # slice off special tokens: [CLS], [SEP]
        tags = tags[: len(input_ids) - 2]
        input_ids = input_ids[1:-1]

        # retrieve slots from the tagged postions and decode slots back
        # into original values
        slots = [token_id for token_id, tag in zip(input_ids, tags) if tag != "o"]
        slots = self._tokenizer.decode(slots)

        # attempt to resolve tagged tokens into slots and
        # collect the successful ones
        slot_meta = self._intent_meta[intent]["slots"]
        parsed_slots = []
        for meta in slot_meta:
            parsed_slots.append(self._parse_slots(meta, slots))

        return {
            "utterance": utterance,
            "intent": intent,
            "confidence": confidence,
            "slots": parsed_slots,
        }

    def _warm_up(self) -> None:
        # make an array the same size as the inputs to warm the
        # model since first inference is always slower than subsequent
        warm = np.zeros((self._model.input_details[0]["shape"]), dtype=np.int32)
        _ = self._model(warm)

    def _encode(self, utterance: str) -> Tuple[np.ndarray, List[int]]:
        inputs = self._tokenizer.encode(utterance)
        # get the non-padded/truncated token ids to match the
        # original utterance to the respective labels and
        # use the length to slice the results
        input_ids = inputs.ids
        # it's (max_length + 1) because the [CLS]
        # token gets appended inside the model
        # notice the slice [1:] when we convert to an array
        inputs.truncate(max_length=self._max_length + 1)
        inputs.pad(length=self._max_length + 1)
        inputs = np.array(inputs.ids[1:], np.int32)
        # add the batch dimension for the TFLite model
        inputs = np.expand_dims(inputs, 0)
        return inputs, input_ids

    def _decode(self, outputs) -> Tuple[str, List[str], float]:
        # to get the index of the highest probability we
        # apply argmax to the posteriors which allows the
        # labels to be decoded with an integer to string mapping
        # we derive the confidence from the highest probability
        intent_posterior, tag_posterior = outputs
        intents = self._decode_intent(intent_posterior)
        tags = self._decode_tags(tag_posterior)
        confidence = np.max(intent_posterior)
        return intents, tags, confidence

    def _decode_tags(self, posterior):
        tags = np.argmax(posterior, -1)[0]
        return [self._tag_decoder.get(tag) for tag in tags]

    def _decode_intent(self, posterior):
        intent = np.argmax(posterior, -1)[0]
        return self._intent_decoder.get(intent)

    def _parse_slots(self, slot_meta, slots):
        slot_type = slot_meta["type"]
        parser = import_module(f"spokestack.nlu.parsers.{slot_type}")
        facets = json.loads(slot_meta["facets"])
        return parser.parse(facets, slots)
