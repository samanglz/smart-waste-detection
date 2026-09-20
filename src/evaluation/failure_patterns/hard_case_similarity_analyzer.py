"""
Hard case similarity analysis for model error investigation.

Compares validation error cases against training objects using
pretrained ResNet50 embeddings.

For each object:
    1. Crop the object using its bounding box.
    2. Extract an embedding from the crop.
    3. Compare validation objects against training objects.
    4. Calculate similarity and class margin.

Validation images are used only as queries and are NEVER added
to the training index.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models

from src.logging.logger import get_logger


logger = get_logger(__name__)


class HardCaseSimilarityAnalyzer:
    """
    Compare validation error objects against training objects
    using pretrained ResNet50 embeddings.

    The comparison is performed on cropped objects rather than
    complete images.
    """

    def __init__(
        self,
        train_images_dir: Path,
        train_labels_dir: Path,
        class_names: List[str],
        device: str = "cuda",
    ):
        self.train_images_dir = Path(train_images_dir)
        self.train_labels_dir = Path(train_labels_dir)

        self.class_names = list(class_names)

        self.class_to_id = {
            name: idx
            for idx, name in enumerate(self.class_names)
        }

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            "Using device: %s",
            self.device,
        )

        # ========================================================
        # ResNet50
        # ========================================================

        weights = models.ResNet50_Weights.DEFAULT

        self.model = models.resnet50(
            weights=weights,
        )

        # Remove classifier.
        self.model.fc = torch.nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        self.transform = weights.transforms()

        # ========================================================
        # Training embedding index
        # ========================================================

        self.train_embeddings: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

    # ============================================================
    # PUBLIC API
    # ============================================================

    def build_train_index(
        self,
        target_classes: List[str],
    ) -> None:
        """
        Extract embeddings for cropped training objects.

        Only objects belonging to target_classes are indexed.
        """

        logger.info(
            "Building train embedding index for classes: %s",
            target_classes,
        )

        self.train_embeddings = {}

        for class_name in target_classes:

            class_id = self._get_class_id(
                class_name
            )

            samples = self._collect_class_samples(
                class_id
            )

            if not samples:
                logger.warning(
                    "No training samples found for class: %s",
                    class_name,
                )
                continue

            embeddings = []

            for sample in samples:

                image = self._load_image(
                    sample["image_path"]
                )

                if image is None:
                    continue

                crop = self._crop_bbox(
                    image,
                    sample["bbox"],
                )

                if crop is None:
                    logger.warning(
                        "Invalid crop: %s",
                        sample["image_path"],
                    )
                    continue

                try:
                    embedding = self._extract_embedding(
                        crop
                    )

                except Exception as exc:
                    logger.error(
                        "Embedding extraction failed for %s: %s",
                        sample["image_path"],
                        exc,
                    )
                    continue

                embeddings.append(
                    {
                        "image_path": sample[
                            "image_path"
                        ],
                        "bbox": sample[
                            "bbox"
                        ],
                        "embedding": embedding,
                    }
                )

            self.train_embeddings[
                class_name
            ] = embeddings

            logger.info(
                "Indexed %d training objects for %s",
                len(embeddings),
                class_name,
            )

    # ============================================================
    # SINGLE CASE
    # ============================================================

    def analyze_case(
        self,
        case: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare one validation error object against
        training object crops.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        image_path = Path(
            case["image_path"]
        )

        image = self._load_image(
            image_path
        )

        if image is None:
            raise ValueError(
                f"Could not load image: {image_path}"
            )

        # ========================================================
        # IMPORTANT:
        # Use GT bbox for the query object.
        # ========================================================

        gt_bbox = case.get(
            "gt_bbox"
        )

        if gt_bbox is None:
            raise ValueError(
                "Error case does not contain gt_bbox."
            )

        crop = self._crop_bbox(
            image,
            gt_bbox,
        )

        if crop is None:
            raise ValueError(
                f"Invalid GT bbox: {gt_bbox}"
            )

        # ========================================================
        # Query embedding
        # ========================================================

        query_embedding = (
            self._extract_embedding(
                crop
            )
        )

        result = {
            "image_path": str(
                image_path
            ),
            "gt_class": case.get(
                "gt_class"
            ),
            "pred_class": case.get(
                "pred_class"
            ),
            "gt_bbox": gt_bbox,
            "pred_bbox": case.get(
                "pred_bbox"
            ),
            "iou": case.get(
                "iou"
            ),
            "confidence": case.get(
                "confidence"
            ),
            "nearest_neighbors": {},
            "best_similarity": {},
            "margin": None,
            "nearest_class": None,
        }

        best_similarities = {}

        # ========================================================
        # Compare against training object crops
        # ========================================================

        for class_name, samples in (
            self.train_embeddings.items()
        ):

            if not samples:
                continue

            neighbors = []

            for sample in samples:

                similarity = (
                    self._cosine_similarity(
                        query_embedding,
                        sample["embedding"],
                    )
                )

                neighbors.append(
                    {
                        "image_path": sample[
                            "image_path"
                        ],
                        "bbox": sample[
                            "bbox"
                        ],
                        "similarity": round(
                            similarity,
                            4,
                        ),
                    }
                )

            neighbors.sort(
                key=lambda x: x[
                    "similarity"
                ],
                reverse=True,
            )

            result[
                "nearest_neighbors"
            ][class_name] = neighbors[
                :top_k
            ]

            best_similarities[
                class_name
            ] = neighbors[0][
                "similarity"
            ]

        # ========================================================
        # Calculate GT vs prediction margin
        # ========================================================

        gt_class = case.get(
            "gt_class"
        )

        pred_class = case.get(
            "pred_class"
        )

        if (
            gt_class in best_similarities
            and pred_class in best_similarities
            and gt_class != pred_class
        ):

            best_gt = best_similarities[
                gt_class
            ]

            best_pred = best_similarities[
                pred_class
            ]

            result[
                "best_similarity"
            ] = {
                gt_class: best_gt,
                pred_class: best_pred,
            }

            margin = (
                best_pred - best_gt
            )

            result["margin"] = round(
                margin,
                4,
            )

            if margin > 0:
                result[
                    "nearest_class"
                ] = pred_class
            else:
                result[
                    "nearest_class"
                ] = gt_class

        return result

    # ============================================================
    # MULTIPLE CASES
    # ============================================================

    def analyze_cases(
        self,
        cases: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple validation error cases.
        """

        results = []

        for i, case in enumerate(cases):

            logger.info(
                "Analyzing case %d/%d",
                i + 1,
                len(cases),
            )

            try:

                result = self.analyze_case(
                    case,
                    top_k=top_k,
                )

                results.append(
                    result
                )

            except Exception as exc:

                logger.error(
                    "Error analyzing case: %s",
                    exc,
                )

        return results
    
    
    def mine_similar_training_objects(
        self,
        cases: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Mine training objects that are most similar to hard
        validation cases.

        For each hard validation case:
            - Use GT bbox to create the query crop.
            - Extract its ResNet50 embedding.
            - Search only training objects belonging to GT class.
            - Keep top-k most similar training objects.

        Validation objects are NEVER added to the training index.

        Returns:
            Dictionary containing:
                - case_results
                - selected_training_objects
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not self.train_embeddings:
            raise RuntimeError(
                "Training embedding index is empty. "
                "Call build_train_index() first."
            )

        case_results = []

        # ------------------------------------------------------------
        # Track how many hard cases selected each training object
        # ------------------------------------------------------------

        selected_objects = {}

        for i, case in enumerate(cases):

            logger.info(
                "Mining hard case %d/%d",
                i + 1,
                len(cases),
            )

            try:

                image_path = Path(
                    case["image_path"]
                )

                image = self._load_image(
                    image_path
                )

                if image is None:
                    logger.warning(
                        "Could not load validation image: %s",
                        image_path,
                    )
                    continue

                gt_bbox = case.get(
                    "gt_bbox"
                )

                if gt_bbox is None:
                    logger.warning(
                        "Missing gt_bbox for: %s",
                        image_path,
                    )
                    continue

                crop = self._crop_bbox(
                    image,
                    gt_bbox,
                )

                if crop is None:
                    logger.warning(
                        "Invalid GT bbox for: %s",
                        image_path,
                    )
                    continue

                # ----------------------------------------------------
                # Query embedding
                # ----------------------------------------------------

                query_embedding = (
                    self._extract_embedding(
                        crop
                    )
                )

                gt_class = case.get(
                    "gt_class"
                )

                if gt_class not in self.train_embeddings:

                    logger.warning(
                        "GT class '%s' is not indexed.",
                        gt_class,
                    )
                    continue

                candidates = []

                # ----------------------------------------------------
                # Compare ONLY against same GT class
                # ----------------------------------------------------

                for sample in self.train_embeddings[
                    gt_class
                ]:

                    similarity = (
                        self._cosine_similarity(
                            query_embedding,
                            sample["embedding"],
                        )
                    )

                    candidates.append(
                        {
                            "image_path": sample[
                                "image_path"
                            ],
                            "bbox": sample[
                                "bbox"
                            ],
                            "class": gt_class,
                            "similarity": round(
                                similarity,
                                4,
                            ),
                        }
                    )

                candidates.sort(
                    key=lambda x: x["similarity"],
                    reverse=True,
                )

                top_candidates = candidates[
                    :top_k
                ]

                # ----------------------------------------------------
                # Register selected training objects
                # ----------------------------------------------------

                for rank, candidate in enumerate(
                    top_candidates,
                    start=1,
                ):

                    key = (
                        candidate["image_path"],
                        tuple(candidate["bbox"]),
                    )

                    if key not in selected_objects:

                        selected_objects[key] = {
                            "image_path": candidate[
                                "image_path"
                            ],
                            "bbox": candidate[
                                "bbox"
                            ],
                            "class": gt_class,
                            "selection_count": 0,
                            "max_similarity": 0.0,
                            "selected_by": [],
                        }

                    selected_objects[key][
                        "selection_count"
                    ] += 1

                    selected_objects[key][
                        "max_similarity"
                    ] = max(
                        selected_objects[key][
                            "max_similarity"
                        ],
                        candidate["similarity"],
                    )

                    selected_objects[key][
                        "selected_by"
                    ].append(
                        {
                            "val_image_path": str(
                                image_path
                            ),
                            "gt_class": gt_class,
                            "pred_class": case.get(
                                "pred_class"
                            ),
                            "iou": case.get(
                                "iou"
                            ),
                            "confidence": case.get(
                                "confidence"
                            ),
                            "rank": rank,
                            "similarity": candidate[
                                "similarity"
                            ],
                        }
                    )

                # ----------------------------------------------------
                # Store case-level result
                # ----------------------------------------------------

                case_results.append(
                    {
                        "val_image_path": str(
                            image_path
                        ),
                        "gt_class": gt_class,
                        "pred_class": case.get(
                            "pred_class"
                        ),
                        "gt_bbox": gt_bbox,
                        "pred_bbox": case.get(
                            "pred_bbox"
                        ),
                        "iou": case.get(
                            "iou"
                        ),
                        "confidence": case.get(
                            "confidence"
                        ),
                        "top_k": top_candidates,
                    }
                )

            except Exception as exc:

                logger.error(
                    "Failed mining case %s: %s",
                    case.get("image_path"),
                    exc,
                )

        # ------------------------------------------------------------
        # Final unique training objects
        # ------------------------------------------------------------

        selected_training_objects = list(
            selected_objects.values()
        )

        selected_training_objects.sort(
            key=lambda x: (
                x["selection_count"],
                x["max_similarity"],
            ),
            reverse=True,
        )

        # Add global rank
        for rank, sample in enumerate(
            selected_training_objects,
            start=1,
        ):
            sample["global_rank"] = rank
            sample["max_similarity"] = round(
                sample["max_similarity"],
                4,
            )

        return {
            "total_hard_cases": len(
                case_results
            ),
            "top_k_per_case": top_k,
            "total_unique_training_objects": len(
                selected_training_objects
            ),
            "case_results": case_results,
            "selected_training_objects": (
                selected_training_objects
            ),
        }


    def save_mining_report(
        self,
        mining_result: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """
        Save hard-case mining results as JSON.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                mining_result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Hard-case mining report saved to: %s",
            output_path,
        )

        return output_path

    # ============================================================
    # REPORT
    # ============================================================

    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path,
    ) -> Path:
        """
        Generate similarity analysis report.
        """

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_cases = len(
            results
        )

        nearest_counts = {}

        for result in results:

            nearest = result.get(
                "nearest_class"
            )

            if nearest:

                nearest_counts[
                    nearest
                ] = (
                    nearest_counts.get(
                        nearest,
                        0,
                    )
                    + 1
                )

        margins = [
            result["margin"]
            for result in results
            if result.get("margin")
            is not None
        ]

        avg_margin = (
            sum(margins)
            / len(margins)
            if margins
            else 0.0
        )

        report = {
            "total_cases": total_cases,
            "nearest_class_counts": (
                nearest_counts
            ),
            "margin_stats": {
                "avg": round(
                    avg_margin,
                    4,
                ),
                "min": (
                    round(
                        min(margins),
                        4,
                    )
                    if margins
                    else None
                ),
                "max": (
                    round(
                        max(margins),
                        4,
                    )
                    if margins
                    else None
                ),
            },
            "results": results,
        }

        report_path = (
            output_dir
            / "similarity_analysis.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        # ========================================================
        # Summary
        # ========================================================

        summary_path = (
            output_dir
            / "summary.txt"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "=" * 60
                + "\n"
            )

            f.write(
                "HARD CASE SIMILARITY ANALYSIS\n"
            )

            f.write(
                "=" * 60
                + "\n\n"
            )

            f.write(
                f"Total cases analyzed: "
                f"{total_cases}\n\n"
            )

            f.write(
                "Nearest class distribution:\n"
            )

            for class_name, count in (
                nearest_counts.items()
            ):

                percentage = (
                    count
                    / total_cases
                    * 100
                    if total_cases
                    else 0.0
                )

                f.write(
                    f"  {class_name}: "
                    f"{count} "
                    f"({percentage:.1f}%)\n"
                )

            f.write(
                f"\nAverage margin "
                f"(pred - gt): "
                f"{avg_margin:.4f}\n"
            )

            f.write(
                "  Positive = closer to predicted class\n"
            )

            f.write(
                "  Negative = closer to ground-truth class\n"
            )

            f.write(
                "\n"
                + "=" * 60
                + "\n"
            )

            f.write(
                "INTERPRETATION:\n"
            )

            f.write(
                "=" * 60
                + "\n"
            )

            if avg_margin > 0.05:

                f.write(
                    "Most hard cases are closer "
                    "to the PREDICTED class in "
                    "embedding space.\n"
                )

                f.write(
                    "This suggests data similarity "
                    "may contribute to the errors.\n"
                )

            elif avg_margin < -0.05:

                f.write(
                    "Most hard cases are closer "
                    "to the GROUND-TRUTH class "
                    "in embedding space.\n"
                )

                f.write(
                    "This suggests model behavior "
                    "may contribute to the errors.\n"
                )

            else:

                f.write(
                    "Hard cases are relatively "
                    "balanced between predicted "
                    "and ground-truth similarity.\n"
                )

                f.write(
                    "Both data similarity and model "
                    "behavior may contribute.\n"
                )

        logger.info(
            "Report saved to: %s",
            report_path,
        )

        return report_path

    # ============================================================
    # TRAINING DATA COLLECTION
    # ============================================================

    def _collect_class_samples(
        self,
        class_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Collect training objects belonging to one class.
        """

        samples = []

        label_files = sorted(
            self.train_labels_dir.glob(
                "*.txt"
            )
        )

        for label_path in label_files:

            image_path = (
                self._find_image(
                    label_path.stem
                )
            )

            if image_path is None:
                continue

            image = self._load_image(
                image_path
            )

            if image is None:
                continue

            height, width = (
                image.shape[:2]
            )

            try:

                with open(
                    label_path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    lines = f.readlines()

            except Exception as exc:

                logger.warning(
                    "Could not read label %s: %s",
                    label_path,
                    exc,
                )

                continue

            for line in lines:

                parts = (
                    line.strip().split()
                )

                if len(parts) != 5:
                    continue

                try:

                    cid = int(
                        parts[0]
                    )

                    if cid != class_id:
                        continue

                    x_center = float(
                        parts[1]
                    )

                    y_center = float(
                        parts[2]
                    )

                    bbox_w = float(
                        parts[3]
                    )

                    bbox_h = float(
                        parts[4]
                    )

                except ValueError:

                    logger.warning(
                        "Invalid label in %s: %s",
                        label_path,
                        line.strip(),
                    )

                    continue

                bbox = (
                    self._yolo_to_xyxy(
                        x_center,
                        y_center,
                        bbox_w,
                        bbox_h,
                        width,
                        height,
                    )
                )

                samples.append(
                    {
                        "image_path": str(
                            image_path
                        ),
                        "bbox": bbox,
                    }
                )

        return samples

    # ============================================================
    # IMAGE
    # ============================================================

    @staticmethod
    def _load_image(
        image_path: Path,
    ):
        """
        Load image as RGB numpy array.
        """

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            return None

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    @staticmethod
    def _crop_bbox(
        image: np.ndarray,
        bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        Crop object from image using xyxy bbox.

        Coordinates are clipped to image boundaries.
        """

        if bbox is None or len(bbox) != 4:
            return None

        height, width = (
            image.shape[:2]
        )

        x1, y1, x2, y2 = map(
            int,
            bbox,
        )

        x1 = max(
            0,
            min(
                x1,
                width - 1,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                height - 1,
            ),
        )

        x2 = max(
            0,
            min(
                x2,
                width,
            ),
        )

        y2 = max(
            0,
            min(
                y2,
                height,
            ),
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            return None

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return crop

    # ============================================================
    # EMBEDDING
    # ============================================================

    @torch.no_grad()
    def _extract_embedding(
        self,
        image,
    ) -> np.ndarray:
        """
        Extract L2-normalized ResNet50 embedding
        from an object crop.
        """

        if isinstance(
            image,
            np.ndarray,
        ):

            image = Image.fromarray(
                image.astype(
                    np.uint8
                )
            )

        elif not isinstance(
            image,
            Image.Image,
        ):

            raise TypeError(
                "Image must be a numpy array "
                "or PIL Image."
            )

        tensor = (
            self.transform(
                image
            )
            .unsqueeze(0)
            .to(self.device)
        )

        embedding = self.model(
            tensor
        )

        embedding = (
            torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1,
            )
        )

        return (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # ============================================================
    # SIMILARITY
    # ============================================================

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity.

        Embeddings are already L2-normalized,
        therefore dot product equals cosine similarity.
        """

        return float(
            np.dot(a, b)
        )

    # ============================================================
    # BBOX
    # ============================================================

    @staticmethod
    def _yolo_to_xyxy(
        x_center: float,
        y_center: float,
        bbox_w: float,
        bbox_h: float,
        img_w: int,
        img_h: int,
    ) -> List[float]:
        """
        Convert normalized YOLO bbox to pixel xyxy.
        """

        x_center *= img_w
        y_center *= img_h

        bbox_w *= img_w
        bbox_h *= img_h

        x1 = (
            x_center
            - bbox_w / 2
        )

        y1 = (
            y_center
            - bbox_h / 2
        )

        x2 = (
            x_center
            + bbox_w / 2
        )

        y2 = (
            y_center
            + bbox_h / 2
        )

        return [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        ]

    # ============================================================
    # HELPERS
    # ============================================================

    def _find_image(
        self,
        stem: str,
    ) -> Optional[Path]:
        """
        Find image by filename stem.
        """

        extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ]

        for ext in extensions:

            image_path = (
                self.train_images_dir
                / f"{stem}{ext}"
            )

            if image_path.exists():
                return image_path

        return None

    def _get_class_id(
        self,
        class_name: str,
    ) -> int:
        """
        Convert class name to class ID.
        """

        if class_name not in self.class_to_id:
            raise ValueError(
                f"Unknown class: {class_name}"
            )

        return self.class_to_id[
            class_name
        ]