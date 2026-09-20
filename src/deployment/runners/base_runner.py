from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseRunner(ABC):
    @property
    @abstractmethod
    def runtime_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    def requested_providers(self) -> list[str] | None:
        return None

    @property
    def actual_providers(self) -> list[str] | None:
        return None

    def prepare_input(self, input_tensor: np.ndarray) -> Any:
        """
        Prepare an input outside the benchmark timing section.

        Runtime-specific runners can override this method to perform
        device transfer, tensor conversion, memory preparation, etc.
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

        return input_tensor

    @abstractmethod
    def predict_tensor(self, input_tensor: Any) -> Any:
        """
        Run inference on an already-prepared input.

        This method is called inside the benchmark timing section.
        Therefore, expensive input preparation or device transfer
        should not be performed here.
        """
        raise NotImplementedError

    def synchronize(self) -> None:
        """
        Synchronize the runtime/device when required.

        CPU and ONNX runners normally do nothing.
        CUDA-based runners should override this method.
        """
        pass

    def metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_name,
            "device": self.device,
            "requested_providers": self.requested_providers,
            "actual_providers": self.actual_providers,
        }