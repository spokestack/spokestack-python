"""Pipeline compatible abstraction for Pytorch jit models."""

import numpy as np
import torch


class PyTorchModel:
    """Pytorch JIT Model."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            inputs = torch.from_numpy(inputs).to(self.device)
            out = self.model(inputs)

        return out
