from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.deployment.runners.onnx_runner import ONNXRunner


class ExportVerifier:
    """
    Verify numerical equivalence between a PyTorch YOLO model
    and its exported ONNX model.

    The same preprocessed input tensor is passed to both models.

    This class performs:
        - input preparation
        - PyTorch inference
        - ONNX inference
        - output shape comparison
        - numerical difference analysis

    It does NOT perform:
        - NMS
        - confidence filtering
        - class decoding
        - mAP evaluation
    """

    def __init__(
        self,
        pytorch_model_path: Path,
        onnx_model_path: Path,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        self.pytorch_model_path = Path(pytorch_model_path)
        self.onnx_model_path = Path(onnx_model_path)
        self.imgsz = imgsz
        self.device = device

        if not self.pytorch_model_path.exists():
            raise FileNotFoundError(
                f"PyTorch model not found: "
                f"{self.pytorch_model_path}"
            )

        if not self.onnx_model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: "
                f"{self.onnx_model_path}"
            )

        # PyTorch / Ultralytics model
        self.pytorch_model = YOLO(
            str(self.pytorch_model_path)
        )

        # ONNX Runtime model
        self.onnx_runner = ONNXRunner(
            model_path=self.onnx_model_path,
            imgsz=self.imgsz,
            providers=[
                "CPUExecutionProvider",
            ],
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_input(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Prepare image in the same tensor format expected by YOLO.

        Input:
            BGR uint8 image, HWC

        Output:
            float32 tensor, BCHW
            shape: (1, 3, imgsz, imgsz)
        """

        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim != 3:
            raise ValueError(
                f"Expected HWC image, got shape {image.shape}"
            )

        image = cv2.resize(
            image,
            (self.imgsz, self.imgsz),
            interpolation=cv2.INTER_LINEAR,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=0,
        )

        return np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # PyTorch inference
    # ------------------------------------------------------------------

    def pytorch_inference(
        self,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Run the YOLO PyTorch model on a preprocessed tensor.

        Returns raw prediction tensor.
        """

        tensor = torch.from_numpy(
            input_tensor
        ).to(self.device)

        model = self.pytorch_model.model
        model.eval()
        model.to(self.device)

        with torch.no_grad():
            output = model(tensor)

        # Ultralytics DetectionModel may return
        # a tensor directly or a tuple/list depending
        # on execution path.
        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Unexpected PyTorch output type: "
                f"{type(output)}"
            )

        return output.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # ONNX inference
    # ------------------------------------------------------------------

    def onnx_inference(
        self,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Run ONNX inference using the exact same input tensor.
        """

        outputs = self.onnx_runner.predict_tensor(
            input_tensor
        )

        if len(outputs) == 0:
            raise RuntimeError(
                "ONNX model returned no outputs."
            )

        return outputs[0]

    # ------------------------------------------------------------------
    # Numerical comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_outputs(
        pytorch_output: np.ndarray,
        onnx_output: np.ndarray,
    ) -> dict:
        """
        Compare PyTorch and ONNX raw outputs.
        """

        if pytorch_output.shape != onnx_output.shape:
            return {
                "shape_match": False,
                "pytorch_shape": pytorch_output.shape,
                "onnx_shape": onnx_output.shape,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "mean_relative_error": None,
            }

        pytorch_output = pytorch_output.astype(
            np.float64
        )

        onnx_output = onnx_output.astype(
            np.float64
        )

        absolute_difference = np.abs(
            pytorch_output - onnx_output
        )

        max_abs_diff = float(
            np.max(absolute_difference)
        )

        mean_abs_diff = float(
            np.mean(absolute_difference)
        )

        denominator = np.maximum(
            np.abs(pytorch_output),
            1e-8,
        )

        relative_error = (
            absolute_difference / denominator
        )

        mean_relative_error = float(
            np.mean(relative_error)
        )

        return {
            "shape_match": True,
            "pytorch_shape": pytorch_output.shape,
            "onnx_shape": onnx_output.shape,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "mean_relative_error": mean_relative_error,
        }

    # ------------------------------------------------------------------
    # Full verification
    # ------------------------------------------------------------------

    def verify(
        self,
        image_path: Path,
    ) -> dict:
        """
        Run the complete PyTorch vs ONNX numerical verification.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        # Important:
        # One identical tensor is used for both models.
        input_tensor = self.prepare_input(
            image
        )

        pytorch_output = self.pytorch_inference(
            input_tensor
        )

        onnx_output = self.onnx_inference(
            input_tensor
        )

        comparison = self.compare_outputs(
            pytorch_output,
            onnx_output,
        )

        return {
            "image": str(image_path),
            "pytorch_model": str(
                self.pytorch_model_path
            ),
            "onnx_model": str(
                self.onnx_model_path
            ),
            "onnx_providers": (
                self.onnx_runner.providers
            ),
            "input_shape": input_tensor.shape,
            **comparison,
        }


def print_report(result: dict) -> None:
    """
    Print a human-readable verification report.
    """

    print()
    print("=" * 72)
    print("PYTORCH vs ONNX - NUMERICAL EQUIVALENCE")
    print("=" * 72)

    print(f"Image           : {result['image']}")
    print(f"PyTorch model   : {result['pytorch_model']}")
    print(f"ONNX model      : {result['onnx_model']}")
    print(f"ONNX providers  : {result['onnx_providers']}")
    print(f"Input shape     : {result['input_shape']}")

    print()
    print("-" * 72)

    print(
        f"Shape match     : {result['shape_match']}"
    )

    print(
        f"PyTorch shape   : {result['pytorch_shape']}"
    )

    print(
        f"ONNX shape      : {result['onnx_shape']}"
    )

    if result["shape_match"]:
        print()
        print(
            f"Max abs diff    : "
            f"{result['max_abs_diff']:.8f}"
        )

        print(
            f"Mean abs diff   : "
            f"{result['mean_abs_diff']:.8f}"
        )

        print(
            f"Mean rel error  : "
            f"{result['mean_relative_error']:.8f}"
        )

    print("=" * 72)