"""
Model export implementation.

Provides a clean interface for exporting trained YOLO models
to deployment formats such as ONNX, TensorRT, TFLite, and TorchScript.
"""

from pathlib import Path
from typing import Optional
from shutil import move

from src.models.yolo.yolo_model import YOLOModel
from src.logging.logger import get_logger
from configs.export_config import EXPORT_FORMAT, SIMPLIFY, DYNAMIC


logger = get_logger(__name__)


class ModelExporter:
    """
    Handles exporting trained YOLO models to deployment formats.

    Default export settings are loaded from configs/export_config.py.
    Per-export overrides are supported.

    Supported formats:
        - ONNX
        - TensorRT
        - TFLite
        - TorchScript
    """

    def __init__(self, model: YOLOModel):
        """
        Initialize the exporter.

        Args:
            model: A trained YOLOModel instance.
        """
        self.model = model

        self.default_format = EXPORT_FORMAT.lower()
        self.default_simplify = SIMPLIFY
        self.default_dynamic = DYNAMIC

        logger.info(
            "ModelExporter initialized | format=%s | simplify=%s | dynamic=%s",
            self.default_format,
            self.default_simplify,
            self.default_dynamic,
        )

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        output_path: Path,
        imgsz: int = 640,
        simplify: Optional[bool] = None,
        dynamic: Optional[bool] = None,
    ) -> Path:

        output_path = self._prepare_output_path(output_path)

        simplify = (
            self.default_simplify
            if simplify is None
            else simplify
        )

        dynamic = (
            self.default_dynamic
            if dynamic is None
            else dynamic
        )

        logger.info("Starting ONNX export...")
        logger.info(f"Input checkpoint: {self.model.weight_path}")
        logger.info(f"Image size: {imgsz}")
        logger.info(f"Simplify: {simplify}")
        logger.info(f"Dynamic: {dynamic}")

        exported_path = self.model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            dynamic=dynamic,
        )

        exported_path = Path(exported_path)

        if not exported_path.exists():
            raise FileNotFoundError(
                f"Ultralytics export reported success, "
                f"but file does not exist: {exported_path}"
            )

        if exported_path.resolve() != output_path.resolve():
            move(
                str(exported_path),
                str(output_path),
            )

        self._verify_export(output_path)

        logger.info(
            f"ONNX export completed successfully: {output_path}"
        )

        return output_path

    def export_default(
        self,
        output_path: Path,
        imgsz: int = 640,
    ) -> Path:

        if self.default_format == "onnx":
            return self.export_onnx(
                output_path=output_path,
                imgsz=imgsz,
            )

        raise ValueError(
            f"Unsupported export format: {self.default_format}"
        )
        
        
    # ------------------------------------------------------------------
    # TensorRT
    # ------------------------------------------------------------------

    def export_tensorrt(
        self,
        output_path: Path,
        imgsz: int = 640,
    ) -> Path:
        """
        Export the model to TensorRT engine format.

        Args:
            output_path: Destination path for the TensorRT engine.
            imgsz: Input image size.

        Returns:
            Path to the exported engine.
        """
        output_path = Path(output_path)

        self._prepare_output_path(output_path)

        logger.info(
            "Starting TensorRT export | output=%s | imgsz=%d",
            output_path,
            imgsz,
        )

        exported_path = self.model.export(
            format="engine",
            imgsz=imgsz,
        )

        exported_path = Path(exported_path)

        self._verify_export(exported_path, "TensorRT")

        logger.info(
            "TensorRT export completed successfully: %s",
            exported_path,
        )

        return exported_path

    # ------------------------------------------------------------------
    # TFLite
    # ------------------------------------------------------------------

    def export_tflite(
        self,
        output_path: Path,
        imgsz: int = 640,
    ) -> Path:
        """
        Export the model to TFLite format.

        Args:
            output_path: Destination path for the TFLite model.
            imgsz: Input image size.

        Returns:
            Path to the exported model.
        """
        output_path = Path(output_path)

        self._prepare_output_path(output_path)

        logger.info(
            "Starting TFLite export | output=%s | imgsz=%d",
            output_path,
            imgsz,
        )

        exported_path = self.model.export(
            format="tflite",
            imgsz=imgsz,
        )

        exported_path = Path(exported_path)

        self._verify_export(exported_path, "TFLite")

        logger.info(
            "TFLite export completed successfully: %s",
            exported_path,
        )

        return exported_path

    # ------------------------------------------------------------------
    # TorchScript
    # ------------------------------------------------------------------

    def export_torchscript(
        self,
        output_path: Path,
        imgsz: int = 640,
    ) -> Path:
        """
        Export the model to TorchScript format.

        Args:
            output_path: Destination path for the TorchScript model.
            imgsz: Input image size.

        Returns:
            Path to the exported model.
        """
        output_path = Path(output_path)

        self._prepare_output_path(output_path)

        logger.info(
            "Starting TorchScript export | output=%s | imgsz=%d",
            output_path,
            imgsz,
        )

        exported_path = self.model.export(
            format="torchscript",
            imgsz=imgsz,
        )

        exported_path = Path(exported_path)

        self._verify_export(exported_path, "TorchScript")

        logger.info(
            "TorchScript export completed successfully: %s",
            exported_path,
        )

        return exported_path

    # ------------------------------------------------------------------
    # Default export
    # ------------------------------------------------------------------

    def export_default(
        self,
        output_path: Path,
        imgsz: int = 640,
    ) -> Path:
        """
        Export using EXPORT_FORMAT from export_config.py.

        Args:
            output_path: Full destination path.
            imgsz: Input image size.

        Returns:
            Path to the exported model.

        Raises:
            ValueError: If the configured format is unsupported.
        """
        output_path = Path(output_path)

        export_methods = {
            "onnx": self.export_onnx,
            "engine": self.export_tensorrt,
            "tensorrt": self.export_tensorrt,
            "tflite": self.export_tflite,
            "torchscript": self.export_torchscript,
        }

        if self.default_format not in export_methods:
            raise ValueError(
                f"Unsupported export format: {self.default_format}. "
                f"Supported formats: {', '.join(export_methods.keys())}"
            )

        logger.info(
            "Running default export | format=%s | output=%s",
            self.default_format,
            output_path,
        )

        export_method = export_methods[self.default_format]

        return export_method(
            output_path=output_path,
            imgsz=imgsz,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_output_path(output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path



    @staticmethod
    def _verify_export(output_path: Path) -> None:
        if not output_path.exists():
            raise FileNotFoundError(
                f"Exported model was not found: {output_path}"
            )

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Exported model is empty: {output_path}"
            )

        logger.info(
            f"Export verified: {output_path} "
            f"({output_path.stat().st_size / (1024 ** 2):.2f} MB)"
        )