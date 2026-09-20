from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.deployment.runners.base_runner import BaseRunner


class PyTorchRunner(BaseRunner):
    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        model_path: str | Path | None = None,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._model_path = (
            Path(model_path)
            if model_path is not None
            else None
        )

        self._model.to(self._device)
        self._model.eval()

    @property
    def runtime_name(self) -> str:
        return "pytorch"

    @property
    def device(self) -> str:
        return str(self._device)

    def prepare_input(
        self,
        input_tensor: np.ndarray,
    ) -> torch.Tensor:
        """
        Prepare the input before benchmark timing.

        NumPy -> PyTorch conversion and device transfer happen here,
        outside the measured inference interval.
        """
        if not isinstance(input_tensor, np.ndarray):
            raise TypeError(
                f"input_tensor must be numpy.ndarray, "
                f"got {type(input_tensor).__name__}"
            )

        if input_tensor.ndim != 4:
            raise ValueError(
                f"input_tensor must be 4D (N, C, H, W), "
                f"got shape {input_tensor.shape}"
            )

        array = np.ascontiguousarray(
            input_tensor,
            dtype=np.float32,
        )

        return torch.from_numpy(array).to(
            device=self._device,
            dtype=torch.float32,
        )

    def predict_tensor(
        self,
        input_tensor: torch.Tensor,
    ) -> Any:
        """
        Run inference on an already-prepared PyTorch tensor.

        No NumPy conversion or device transfer happens here.
        This method is intended to be called inside benchmark timing.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(
                f"input_tensor must be torch.Tensor, "
                f"got {type(input_tensor).__name__}"
            )

        if input_tensor.ndim != 4:
            raise ValueError(
                f"input_tensor must be 4D (N, C, H, W), "
                f"got shape {tuple(input_tensor.shape)}"
            )

        if input_tensor.device.type != self._device.type:
            raise ValueError(
                f"Input tensor is on {input_tensor.device}, "
                f"but runner expects {self._device}."
            )

        if self._device.type == "cuda":
            expected_index = (
                torch.cuda.current_device()
                if self._device.index is None
                else self._device.index
            )

            actual_index = (
                torch.cuda.current_device()
                if input_tensor.device.index is None
                else input_tensor.device.index
            )

            if actual_index != expected_index:
                raise ValueError(
                    f"Input tensor is on {input_tensor.device}, "
                    f"but runner expects {self._device}."
                )
                    
        with torch.inference_mode():
            return self._model(input_tensor)

    def synchronize(self) -> None:
        """
        Synchronize CUDA before/after timed inference.
        """
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()

        metadata.update(
            {
                "model_path": (
                    str(self._model_path.resolve())
                    if self._model_path is not None
                    else None
                ),
                "torch_device": str(self._device),
                "dtype": "float32",
            }
        )

        return metadata