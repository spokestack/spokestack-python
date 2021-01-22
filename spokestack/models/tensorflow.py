"""
TFLite model base class
"""
from typing import Any, List

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow.lite as tflite


class TFLiteModel:
    """TFLite model base class for managing multiple inputs/outputs

    Args:
        model_path (str): Path to .tflite model file
        **kwargs (Any): Additional keywords arguments for the TFLite Interpreter.
                        [https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter]
    """

    def __init__(self, model_path: str, **kwargs: Any) -> None:

        self._interpreter: tflite.Interpreter = tflite.Interpreter(
            model_path=model_path, **kwargs
        )
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._interpreter.allocate_tensors()

    def __call__(self, *args: Any) -> List[np.ndarray]:
        """Forward pass of the TFLite model

        Args:
            inputs (Any): inputs to the TFLite model

        Returns: outputs of TFLite model

        """

        for detail, arg in zip(self._input_details, args):
            self._interpreter.set_tensor(detail["index"], arg)

        self._interpreter.invoke()

        return [
            self._interpreter.get_tensor(tensor["index"])
            for tensor in self._output_details
        ]

    @property
    def input_details(self) -> List[Any]:
        """Property for accesing the TFLite model input_details

        Returns: Input details for the TFLite model

        """
        return self._input_details

    @property
    def output_details(self) -> List[Any]:
        """Property for accesing the TFLite model output_details

        Returns: Output details for the TFLite model

        """
        return self._output_details
