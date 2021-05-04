import json
from unittest import mock

import numpy as np

from spokestack.tts.lite import SpeechSynthesizer, BLOCK_LENGTH, BLOCK_OVERLAP

ALPHABET = "_^~abcdefghijklmnopqrstuvwxyzæðŋɑɔəɛɝɪʃʊʌʒˈˌːθɡxyɹʰɜɒɚɱʔɨɾɐʁɵχ "
LEXICON = """
in\tɪn
desert\tdɪˈzɝːt\tVBP
desert\tˈdɛzɝt
"""


class ModelFactory(mock.MagicMock):
    def __call__(self, model_path):
        model = mock.MagicMock()
        if model_path.endswith("align.tflite"):
            model.input_details = [{"index": 0}]
            model.return_value = [np.zeros([100, 256], dtype=np.float32)]
        elif model_path.endswith("encode.tflite"):
            model.input_details = [{"index": 0}]
            model.return_value = [np.zeros([100, 80], dtype=np.float32)]
        elif model_path.endswith("decode.tflite"):
            model.input_details = [{"index": 0}]
            model.return_value = [
                np.zeros([BLOCK_LENGTH + BLOCK_OVERLAP], dtype=np.float32)
            ]
        return model


@mock.patch("spokestack.tts.lite.TFLiteModel", new_callable=ModelFactory)
def test_synthesizer(_mock, tmpdir):
    with open(tmpdir / "lexicon.txt", "w") as file:
        file.write(LEXICON)

    with open(tmpdir / "metadata.json", "w") as file:
        json.dump({"language": "en", "alphabet": list(ALPHABET)}, file)

    synth = SpeechSynthesizer(tmpdir)

    # basic synth smoke test
    blocks = list(synth.synthesize("I desert in the desert."))
    assert len(blocks) == 3
    for block in blocks:
        assert len(block) <= BLOCK_LENGTH

    # sentence segmentation
    blocks = list(synth.synthesize("This is a test. This is another one."))
    assert len(blocks) == 6
    for block in blocks:
        assert len(block) <= BLOCK_LENGTH
