from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from src.deployment.runners.onnx_runner import ONNXRunner


class PredictionComparator:
    """
    Compare final object detections between a PyTorch YOLO model
    and its ONNX export.

    Both models receive the exact same preprocessed tensor and
    use the same Ultralytics NMS implementation.

    Comparison includes:
        - detection count
        - class IDs
        - confidence
        - bounding boxes
        - IoU between matched boxes
    """

    def __init__(
        self,
        pytorch_model_path: Path,
        onnx_model_path: Path,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.pytorch_model_path = Path(pytorch_model_path)
        self.onnx_model_path = Path(onnx_model_path)

        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
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

        self.pytorch_model = YOLO(
            str(self.pytorch_model_path)
        )

        self.pytorch_model.model.eval()
        self.pytorch_model.model.to(self.device)

        self.onnx_runner = ONNXRunner(
            model_path=self.onnx_model_path,
            imgsz=self.imgsz,
            providers=["CPUExecutionProvider"],
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: int = 640,
        color: tuple[int, int, int] = (114, 114, 114),
    ):
        """
        Resize image while preserving aspect ratio and pad to a
        fixed square size.

        Returns:
            padded image
            scale ratio
            padding (dw, dh)
        """

        shape = image.shape[:2]  # (height, width)

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        ratio = min(
            new_shape[0] / shape[0],
            new_shape[1] / shape[1],
        )

        new_unpad = (
            int(round(shape[1] * ratio)),
            int(round(shape[0] * ratio)),
        )

        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(
                image,
                new_unpad,
                interpolation=cv2.INTER_LINEAR,
            )

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )

        return image, ratio, (dw, dh)

    def preprocess(
        self,
        image: np.ndarray,
    ):
        """
        Prepare image for both PyTorch and ONNX.

        Both models receive exactly the same tensor.

        Returns:
            input_tensor
            original_shape
            ratio
            padding
        """

        if image is None:
            raise ValueError("Input image is None.")

        original_shape = image.shape[:2]

        image, ratio, padding = self.letterbox(
            image,
            new_shape=self.imgsz,
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

        image = np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

        return (
            image,
            original_shape,
            ratio,
            padding,
        )

    # ------------------------------------------------------------------
    # PyTorch
    # ------------------------------------------------------------------

    def run_pytorch(
        self,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Run raw PyTorch inference.
        """

        tensor = torch.from_numpy(
            input_tensor
        ).to(self.device)

        with torch.no_grad():
            output = self.pytorch_model.model(
                tensor
            )

        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Unexpected PyTorch output type: "
                f"{type(output)}"
            )

        return output.detach().cpu()

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def run_onnx(
        self,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        """
        Run raw ONNX inference.
        """

        outputs = self.onnx_runner.predict_tensor(
            input_tensor
        )

        if not outputs:
            raise RuntimeError(
                "ONNX returned no outputs."
            )

        return outputs[0]

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    def postprocess(
        self,
        prediction,
        original_shape,
        input_shape,
    ) -> np.ndarray:
        """
        Apply the same Ultralytics NMS to predictions.

        Returns:
            ndarray with columns:
            [x1, y1, x2, y2, confidence, class_id]
        """

        if isinstance(prediction, np.ndarray):
            prediction = torch.from_numpy(
                prediction
            )

        if prediction.ndim != 3:
            raise ValueError(
                f"Expected prediction with 3 dimensions, "
                f"got {prediction.shape}"
            )

        detections = non_max_suppression(
            prediction,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=300,
        )[0]

        if detections.numel() == 0:
            return np.empty(
                (0, 6),
                dtype=np.float32,
            )

        # Convert boxes from 640x640 letterboxed coordinates
        # back to original image coordinates.
        detections[:, :4] = scale_boxes(
            input_shape,
            detections[:, :4],
            original_shape,
        )

        return detections.cpu().numpy()

    # ------------------------------------------------------------------
    # IoU
    # ------------------------------------------------------------------

    @staticmethod
    def box_iou(
        box1: np.ndarray,
        box2: np.ndarray,
    ) -> float:
        """
        Calculate IoU between two xyxy boxes.
        """

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area1 = max(
            0.0,
            box1[2] - box1[0],
        ) * max(
            0.0,
            box1[3] - box1[1],
        )

        area2 = max(
            0.0,
            box2[2] - box2[0],
        ) * max(
            0.0,
            box2[3] - box2[1],
        )

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return float(
            intersection / union
        )

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_detections(
        self,
        pytorch_detections: np.ndarray,
        onnx_detections: np.ndarray,
        iou_match_threshold: float = 0.5,
    ):
        """
        Match PyTorch and ONNX detections.

        Matching requires:
            1. Same class
            2. IoU >= threshold

        Each detection can only be matched once.
        """

        matches = []
        used_onnx = set()

        for pt_index, pt_det in enumerate(
            pytorch_detections
        ):
            best_match = None
            best_iou = 0.0

            pt_class = int(pt_det[5])

            for onnx_index, onnx_det in enumerate(
                onnx_detections
            ):
                if onnx_index in used_onnx:
                    continue

                onnx_class = int(onnx_det[5])

                if pt_class != onnx_class:
                    continue

                iou = self.box_iou(
                    pt_det[:4],
                    onnx_det[:4],
                )

                if (
                    iou >= iou_match_threshold
                    and iou > best_iou
                ):
                    best_iou = iou
                    best_match = onnx_index

            if best_match is not None:
                used_onnx.add(best_match)

                matches.append(
                    {
                        "pytorch_index": pt_index,
                        "onnx_index": best_match,
                        "class_id": pt_class,
                        "pytorch_conf": float(
                            pt_det[4]
                        ),
                        "onnx_conf": float(
                            onnx_detections[
                                best_match
                            ][4]
                        ),
                        "iou": best_iou,
                    }
                )

        return matches

    # ------------------------------------------------------------------
    # Full comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        image_path: Path,
    ) -> dict:
        """
        Run complete prediction-level comparison.
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

        (
            input_tensor,
            original_shape,
            _ratio,
            _padding,
        ) = self.preprocess(image)

        pytorch_raw = self.run_pytorch(
            input_tensor
        )

        onnx_raw = self.run_onnx(
            input_tensor
        )

        pytorch_detections = self.postprocess(
            pytorch_raw,
            original_shape,
            input_tensor.shape[2:],
        )

        onnx_detections = self.postprocess(
            onnx_raw,
            original_shape,
            input_tensor.shape[2:],
        )

        matches = self.match_detections(
            pytorch_detections,
            onnx_detections,
        )

        unmatched_pytorch = (
            len(pytorch_detections)
            - len(matches)
        )

        unmatched_onnx = (
            len(onnx_detections)
            - len(matches)
        )

        mean_iou = (
            float(
                np.mean(
                    [
                        match["iou"]
                        for match in matches
                    ]
                )
            )
            if matches
            else 0.0
        )

        mean_conf_difference = (
            float(
                np.mean(
                    [
                        abs(
                            match["pytorch_conf"]
                            - match["onnx_conf"]
                        )
                        for match in matches
                    ]
                )
            )
            if matches
            else 0.0
        )

        return {
            "image": str(image_path),
            "pytorch_model": str(
                self.pytorch_model_path
            ),
            "onnx_model": str(
                self.onnx_model_path
            ),
            "pytorch_count": len(
                pytorch_detections
            ),
            "onnx_count": len(
                onnx_detections
            ),
            "matched_count": len(matches),
            "unmatched_pytorch": unmatched_pytorch,
            "unmatched_onnx": unmatched_onnx,
            "mean_iou": mean_iou,
            "mean_conf_difference": (
                mean_conf_difference
            ),
            "pytorch_detections": (
                pytorch_detections
            ),
            "onnx_detections": (
                onnx_detections
            ),
            "matches": matches,
        }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_report(result: dict) -> None:
    """
    Print a human-readable comparison report.
    """

    print()
    print("=" * 72)
    print("PYTORCH vs ONNX - PREDICTION COMPARISON")
    print("=" * 72)

    print(
        f"Image             : "
        f"{result['image']}"
    )

    print(
        f"PyTorch model     : "
        f"{result['pytorch_model']}"
    )

    print(
        f"ONNX model        : "
        f"{result['onnx_model']}"
    )

    print()
    print("-" * 72)

    print(
        f"PyTorch detections: "
        f"{result['pytorch_count']}"
    )

    print(
        f"ONNX detections   : "
        f"{result['onnx_count']}"
    )

    print(
        f"Matched           : "
        f"{result['matched_count']}"
    )

    print(
        f"Unmatched PyTorch : "
        f"{result['unmatched_pytorch']}"
    )

    print(
        f"Unmatched ONNX    : "
        f"{result['unmatched_onnx']}"
    )

    print(
        f"Mean matched IoU  : "
        f"{result['mean_iou']:.6f}"
    )

    print(
        f"Mean conf diff    : "
        f"{result['mean_conf_difference']:.8f}"
    )

    print()
    print("-" * 72)
    print("MATCHED DETECTIONS")
    print("-" * 72)

    if not result["matches"]:
        print("No matched detections.")

    for match in result["matches"]:
        print(
            f"class={match['class_id']} | "
            f"PyTorch conf={match['pytorch_conf']:.6f} | "
            f"ONNX conf={match['onnx_conf']:.6f} | "
            f"IoU={match['iou']:.6f}"
        )

    print()
    print("=" * 72)     