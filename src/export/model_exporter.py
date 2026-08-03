"""
Model export implementation.

This module provides a clean interface for exporting trained YOLO models
to various deployment formats such as ONNX, TensorRT, TFLite, and TorchScript.
"""

from pathlib import Path

from src.models.yolo.yolo_model import YOLOModel
from src.logging.logger import get_logger
from configs.export_config import EXPORT_FORMAT, SIMPLIFY, DYNAMIC


logger = get_logger(__name__)


class ModelExporter:
    """
    Handles exporting trained YOLO models to deployment formats.

    Uses settings from configs/export_config.py as defaults, while allowing
    per-method overrides for flexibility.

    Supported formats:
        - ONNX: Standard cross-platform format
        - TensorRT: Optimized for NVIDIA GPUs
        - TFLite: Optimized for mobile and edge devices
        - TorchScript: For C++ deployment
    """

    def __init__(self, model: YOLOModel):
        """
        Initialize exporter with a trained model.

        Args:
            model: Trained YOLOModel instance.
        """
        self.model = model
        self.default_format = EXPORT_FORMAT
        self.default_simplify = SIMPLIFY
        self.default_dynamic = DYNAMIC
        logger.info("ModelExporter initialized with default format: %s", self.default_format)

    def export_onnx(
        self,
        output_path: Path,
        imgsz: int = 640,
        simplify: bool = None,
        dynamic: bool = None,
    ) -> Path:
        """
        Export model to ONNX format.

        If simplify or dynamic are not provided, values are taken from config.

        Args:
            output_path: Path where the ONNX file will be saved.
            imgsz: Input image size (default: 640).
            simplify: Whether to simplify the ONNX graph (default from config).
            dynamic: Whether to allow dynamic input sizes (default from config).

        Returns:
            The output path where the model was saved.

        Raises:
            Exception: If export fails.
        """
        if simplify is None:
            simplify = self.default_simplify
        if dynamic is None:
            dynamic = self.default_dynamic

        logger.info(
            "Exporting to ONNX: %s (imgsz=%d, simplify=%s, dynamic=%s)",
            output_path, imgsz, simplify, dynamic
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.export(
            format="onnx",
            path=str(output_path),   # ✅ crucial: specify output path
            imgsz=imgsz,
            simplify=simplify,
            dynamic=dynamic,
        )

        logger.info("ONNX export complete.")
        return output_path

    def export_tensorrt(self, output_path: Path, imgsz: int = 640) -> Path:
        """
        Export model to TensorRT engine format.

        TensorRT provides the fastest inference speed on NVIDIA GPUs.

        Args:
            output_path: Path where the .engine file will be saved.
            imgsz: Input image size (default: 640).

        Returns:
            The output path where the model was saved.
        """
        logger.info("Exporting to TensorRT: %s (imgsz=%d)", output_path, imgsz)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.export(
            format="engine",
            path=str(output_path),   # ✅ specify output path
            imgsz=imgsz,
        )

        logger.info("TensorRT export complete.")
        return output_path

    def export_tflite(self, output_path: Path, imgsz: int = 640) -> Path:
        """
        Export model to TFLite format for mobile and edge deployment.

        Args:
            output_path: Path where the .tflite file will be saved.
            imgsz: Input image size (default: 640).

        Returns:
            The output path where the model was saved.
        """
        logger.info("Exporting to TFLite: %s (imgsz=%d)", output_path, imgsz)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.export(
            format="tflite",
            path=str(output_path),   # ✅ specify output path
            imgsz=imgsz,
        )

        logger.info("TFLite export complete.")
        return output_path

    def export_torchscript(self, output_path: Path, imgsz: int = 640) -> Path:
        """
        Export model to TorchScript format for C++ deployment.

        Args:
            output_path: Path where the TorchScript file will be saved.
            imgsz: Input image size (default: 640).

        Returns:
            The output path where the model was saved.
        """
        logger.info("Exporting to TorchScript: %s (imgsz=%d)", output_path, imgsz)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.export(
            format="torchscript",
            path=str(output_path),   # ✅ specify output path
            imgsz=imgsz,
        )

        logger.info("TorchScript export complete.")
        return output_path

    def export_default(self, output_dir: Path, imgsz: int = 640) -> Path:
        """
        Export model using the default format specified in config.

        The output file name is automatically set to "model.{ext}" where
        ext is determined by the format (e.g., .onnx, .engine, etc.).

        Args:
            output_dir: Directory where the exported file will be saved.
            imgsz: Input image size (default: 640).

        Returns:
            The full path to the exported file.

        Raises:
            ValueError: If the default format is unsupported.
        """
        format_map = {
            "onnx": ("model.onnx", self.export_onnx),
            "engine": ("model.engine", self.export_tensorrt),
            "tflite": ("model.tflite", self.export_tflite),
            "torchscript": ("model.torchscript", self.export_torchscript),
        }

        if self.default_format not in format_map:
            raise ValueError(f"Unsupported export format: {self.default_format}")

        filename, export_method = format_map[self.default_format]
        output_path = output_dir / filename

        logger.info("Exporting with default format (%s) to: %s", self.default_format, output_path)
        return export_method(output_path, imgsz)