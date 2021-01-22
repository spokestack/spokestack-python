"""
This module contains the class for using TFLite NLU models. In this case, an NLU model
is a TFLite model which takes in an utterance and returns an intent along with
any slots that are associated with that intent.
"""
import json
import logging
import os
from importlib import import_module
from typing import Any, Dict, List, Tuple

import numpy as np
from tokenizers import BertWordPieceTokenizer

from spokestack import utils
from spokestack.models.tensorflow import TFLiteModel
from spokestack.nlu.result import Result

_LOG = logging.getLogger(__name__)


class TFLiteNLU:
    """Abstraction for using TFLite NLU models

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
        self._slot_meta = {}
        for intent in self._intent_meta:
            for slot in self._intent_meta[intent]["slots"]:
                self._slot_meta[slot.pop("name")] = slot
        self._warm_up()

    def __call__(self, utterance: str) -> Result:
        """Classifies a string utterance into an intent and identifies any associated
            slots contained in the utterance. The slots get parsed based on type and
            then returned along with the intent and its associated confidence value.

        Args:
            utterance (str): string that needs to be understood

        Returns (Result): A class with properties for the identified intent, along with
                        raw, parsed slots and model confidence in prediction

        """
        inputs, input_ids = self._encode(utterance)
        outputs = self._model(inputs)
        intent, tags, confidence = self._decode(outputs)

        # slice off special tokens: [CLS], [SEP]
        tags = tags[: len(input_ids) - 2]
        _LOG.debug(f"{tags}")
        input_ids = input_ids[1:-1]
        _LOG.debug(f"{input_ids}")
        # retrieve slots from the tagged positions and decode slots back
        # into original values
        slots = [
            (token_id, tag[2:]) for token_id, tag in zip(input_ids, tags) if tag != "o"
        ]
        _LOG.debug(f"{slots}")

        slot_map: dict = {}
        for (token, tag) in slots:
            if tag in slot_map:
                slot_map[tag].append(token)
            else:
                slot_map[tag] = [token]

        for key, value in slot_map.items():
            slot_map[key] = self._tokenizer.decode(value)

        # attempt to resolve tagged tokens into slots and
        # collect the successful ones
        parsed_slots = {}
        for key in slot_map:
            parsed = self._parse_slots(self._slot_meta[key], slot_map[key])
            parsed_slots[key] = {
                "name": key,
                "parsed_value": parsed,
                "raw_value": slot_map[key],
            }
        _LOG.debug(f"parsed slots: {parsed_slots}")
        return Result(
            utterance=utterance,
            intent=intent,
            confidence=confidence,
            slots=parsed_slots,
        )

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

    def _decode(self, outputs: list) -> Tuple[str, List[str], float]:
        # to get the index of the highest probability we
        # apply argmax to the posteriors which allows the
        # labels to be decoded with an integer to string mapping
        # we derive the confidence from the highest probability
        intent_posterior, tag_posterior = outputs
        intents, confidence = self._decode_intent(intent_posterior)
        tags = self._decode_tags(tag_posterior)
        _LOG.debug(f"decoded tags: {tags}")
        _LOG.debug(f"decoded intent: {intents}")
        _LOG.debug(f"confidence: {confidence}")
        return intents, tags, confidence

    def _decode_tags(self, posterior: np.ndarray) -> List[Any]:
        posterior = np.squeeze(posterior, 0)
        tags = np.argmax(posterior, -1)
        return [self._tag_decoder.get(tag) for tag in tags]

    def _decode_intent(self, posterior: np.ndarray) -> Any:
        posterior = np.squeeze(posterior, 0)
        intent = np.argmax(posterior, -1)
        return self._intent_decoder.get(intent), posterior[intent]

    def _parse_slots(self, slot_meta: Dict[str, Any], slots: Dict[str, Any]) -> Any:
        slot_type = slot_meta["type"]
        parser = import_module(f"spokestack.nlu.parsers.{slot_type}")
        facets = json.loads(slot_meta["facets"])
        return parser.parse(facets, slots)  # type: ignore
