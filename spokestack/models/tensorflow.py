"""
TFLite model base class
"""
from typing import Any, List

import numpy as np  # type: ignore
import tflite_runtime.interpreter as tflite  # type: ignore


class TFLiteModel:
    """ TFLite model base class for managing multiple inputs/outputs

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

    def __call__(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """ Foward pass of the TFLite model

        Args:
            inputs (List[np.ndarray]): inputs to the TFLite model

        Returns: outputs of TFLite model

        """

        for detail, i in zip(self.input_details, inputs):
            self._interpreter.set_tensor(detail["index"], i)

        self._interpreter.invoke()

        return [
            self._interpreter.get_tensor(tensor["index"])
            for tensor in self._output_details
        ]

    @property
    def input_details(self) -> List[Any]:
        """ Property for accesing the TFLite model input_details

        Returns: Input details for the TFLite model

        """
        return self._input_details

    @property
    def output_details(self) -> List[Any]:
        """ Property for accesing the TFLite model output_details

        Returns: Output details for the TFLite model

        """
        return self._output_details

    def reset(self) -> None:
        """ Resets the variables for the TFLite model """
        self._interpreter.reset_all_variables()
