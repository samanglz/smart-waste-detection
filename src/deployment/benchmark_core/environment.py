from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EnvironmentInfo:
    """
    Runtime environment information captured for benchmark
    reproducibility.
    """

    python_version: str
    os: str
    platform: str

    pytorch_version: str | None
    onnxruntime_version: str | None

    cuda_available: bool
    cuda_version: str | None

    gpu_name: str | None
    gpu_memory_mb: int | None

    onnx_available_providers: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def collect_environment_info() -> EnvironmentInfo:
    """
    Collect information about the current benchmark environment.

    Optional dependencies such as PyTorch and ONNX Runtime are handled
    gracefully so that this module remains runtime-agnostic.
    """

    pytorch_version: str | None = None
    cuda_available = False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None

    try:
        import torch

        pytorch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())

        if cuda_available:
            cuda_version = torch.version.cuda

            device_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_index)

            total_memory = torch.cuda.get_device_properties(
                device_index
            ).total_memory

            gpu_memory_mb = int(
                total_memory / (1024 * 1024)
            )

    except ImportError:
        pass
    except Exception:
        # Environment collection must never crash the benchmark
        # because of an optional hardware/runtime query.
        pass

    onnxruntime_version: str | None = None
    onnx_available_providers: list[str] = []

    try:
        import onnxruntime as ort

        onnxruntime_version = ort.__version__
        onnx_available_providers = list(
            ort.get_available_providers()
        )

    except ImportError:
        pass
    except Exception:
        pass

    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        os=platform.system(),
        platform=platform.platform(),
        pytorch_version=pytorch_version,
        onnxruntime_version=onnxruntime_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        onnx_available_providers=onnx_available_providers,
    )