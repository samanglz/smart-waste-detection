
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from src.evaluation.failure_patterns.failure_pattern import (
    FailurePattern,
)
from src.evaluation.failure_patterns.embedding_extractor import (
    EmbeddingExtractor,
)
from src.evaluation.failure_patterns.embedding_similarity import (
    EmbeddingSimilarity,
)
from src.logging.logger import get_logger


logger = get_logger(__name__)


class FailurePatternMiner:
    """
    Find training samples related to discovered failure patterns.

    Architecture
    ------------
    Validation set:
        Model errors
            ↓
        FailurePatternAnalyzer
            ↓
        FailurePattern

    Training set:
        Training objects
            ↓
        Embeddings
            ↓
        Similarity search
            ↓
        Candidate training samples

    IMPORTANT
    ---------
    Validation samples are never returned as training candidates.

    This class does NOT:
        - modify the dataset
        - augment images
        - train the model
        - modify validation/test data
    """

    def __init__(
        self,
        train_images_dir: Path,
        train_labels_dir: Path,
        class_names: List[str],
        embedding_extractor: EmbeddingExtractor,
    ):
        self.train_images_dir = Path(
            train_images_dir
        )

        self.train_labels_dir = Path(
            train_labels_dir
        )

        self.class_names = list(
            class_names
        )

        self.class_to_id = {
            name: idx
            for idx, name in enumerate(
                self.class_names
            )
        }

        self.embedding_extractor = (
            embedding_extractor
        )

        # --------------------------------------------------------
        # Training embedding index
        # --------------------------------------------------------

        self._train_index: List[Dict[str, Any]] = []

        self._index_built = False

    # ============================================================
    # PUBLIC API
    # ============================================================

    def build_index(
        self,
        target_classes: List[str] | None = None,
    ) -> int:
        """
        Build embedding index from training objects.

        Args:
            target_classes:
                Optional list of classes to index.

                If None, all classes are indexed.

        Returns:
            Number of indexed training objects.
        """

        if target_classes is None:
            target_classes = self.class_names

        target_class_ids = {
            self._get_class_id(name)
            for name in target_classes
        }

        logger.info(
            "Building training embedding index "
            "for classes: %s",
            target_classes,
        )

        self._train_index = []

        label_files = sorted(
            self.train_labels_dir.glob("*.txt")
        )

        for label_path in label_files:

            image_path = self._find_image(
                label_path.stem
            )

            if image_path is None:
                logger.warning(
                    "Image not found for label: %s",
                    label_path,
                )
                continue

            image_width, image_height = (
                self._get_image_size(
                    image_path
                )
            )

            if image_width <= 0 or image_height <= 0:
                continue

            try:
                with open(
                    label_path,
                    "r",
                    encoding="utf-8",
                ) as file:

                    lines = file.readlines()

            except Exception as exc:
                logger.error(
                    "Failed to read label %s: %s",
                    label_path,
                    exc,
                )
                continue

            # ----------------------------------------------------
            # Process every object in image
            # ----------------------------------------------------

            for object_index, line in enumerate(lines):

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])

                    if class_id not in target_class_ids:
                        continue

                    cx = float(parts[1])
                    cy = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                except ValueError:
                    logger.warning(
                        "Invalid label line in %s: %s",
                        label_path,
                        line.strip(),
                    )
                    continue

                bbox = self._yolo_to_xyxy(
                    cx,
                    cy,
                    width,
                    height,
                    image_width,
                    image_height,
                )

                try:
                    embedding = (
                        self.embedding_extractor.extract(
                            image_path
                        )
                    )

                except Exception as exc:
                    logger.error(
                        "Embedding extraction failed "
                        "for %s: %s",
                        image_path,
                        exc,
                    )
                    continue

                self._train_index.append(
                    {
                        "image_path": str(
                            image_path
                        ),
                        "label_path": str(
                            label_path
                        ),
                        "object_index": object_index,
                        "class_id": class_id,
                        "class_name": self.class_names[
                            class_id
                        ],
                        "bbox": bbox,
                        "embedding": embedding,
                    }
                )

        self._index_built = True

        logger.info(
            "Training embedding index built: "
            "%d objects",
            len(self._train_index),
        )

        return len(self._train_index)

    # ============================================================
    # PATTERN MINING
    # ============================================================

    def mine_pattern(
        self,
        pattern: FailurePattern,
        query_cases: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Mine training samples related to one failure pattern.

        Args:
            pattern:
                Failure pattern discovered from validation.

            query_cases:
                Validation error cases associated with this pattern.

            top_k:
                Number of nearest training samples per query.

        Returns:
            Mining result.
        """

        if not self._index_built:
            raise RuntimeError(
                "Training embedding index has not been built. "
                "Call build_index() first."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        results = []

        for case in query_cases:

            query_embedding = (
                self._extract_query_embedding(
                    case
                )
            )

            if query_embedding is None:
                continue

            candidates = (
                self._filter_candidates(
                    pattern
                )
            )

            if not candidates:
                continue

            candidate_embeddings = np.stack(
                [
                    item["embedding"]
                    for item in candidates
                ]
            )

            indices, similarities = (
                EmbeddingSimilarity.top_k(
                    query_embedding,
                    candidate_embeddings,
                    k=top_k,
                )
            )

            neighbors = []

            for index, similarity in zip(
                indices,
                similarities,
            ):

                candidate = candidates[
                    int(index)
                ]

                neighbors.append(
                    {
                        "image_path": candidate[
                            "image_path"
                        ],
                        "label_path": candidate[
                            "label_path"
                        ],
                        "object_index": candidate[
                            "object_index"
                        ],
                        "class_name": candidate[
                            "class_name"
                        ],
                        "bbox": candidate[
                            "bbox"
                        ],
                        "similarity": round(
                            float(similarity),
                            4,
                        ),
                    }
                )

            results.append(
                {
                    "validation_image": case.get(
                        "image_path"
                    ),
                    "gt_class": case.get(
                        "gt_class"
                    ),
                    "pred_class": case.get(
                        "pred_class"
                    ),
                    "iou": case.get(
                        "iou"
                    ),
                    "confidence": case.get(
                        "confidence"
                    ),
                    "neighbors": neighbors,
                }
            )

        return {
            "pattern_id": pattern.pattern_id,
            "error_type": pattern.error_type,
            "source_class": pattern.source_class,
            "target_class": pattern.target_class,
            "num_queries": len(
                query_cases
            ),
            "results": results,
        }

    # ============================================================
    # CANDIDATE FILTERING
    # ============================================================

    def _filter_candidates(
        self,
        pattern: FailurePattern,
    ) -> List[Dict[str, Any]]:
        """
        Select training objects relevant to a failure pattern.

        For confusion:
            Prefer both source and target classes.

        For FN:
            Use source class.

        For FP:
            Use source class.
        """

        if pattern.error_type == "confusion":

            allowed_classes = {
                pattern.source_class,
                pattern.target_class,
            }

        else:

            allowed_classes = {
                pattern.source_class
            }

        return [
            item
            for item in self._train_index
            if item["class_name"]
            in allowed_classes
        ]

    # ============================================================
    # QUERY EMBEDDING
    # ============================================================

    def _extract_query_embedding(
        self,
        case: Dict[str, Any],
    ) -> np.ndarray | None:
        """
        Extract embedding for a validation error case.

        The validation image itself is used only as a query.
        """

        image_path = case.get(
            "image_path"
        )

        if image_path is None:
            logger.warning(
                "Query case has no image_path."
            )
            return None

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            logger.warning(
                "Validation image not found: %s",
                image_path,
            )
            return None

        return self.embedding_extractor.extract(
            image_path
        )

    # ============================================================
    # HELPERS
    # ============================================================

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

    def _find_image(
        self,
        stem: str,
    ) -> Path | None:
        """
        Find training image by filename stem.
        """

        extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        )

        for extension in extensions:

            image_path = (
                self.train_images_dir
                / f"{stem}{extension}"
            )

            if image_path.exists():
                return image_path

        return None

    @staticmethod
    def _get_image_size(
        image_path: Path,
    ) -> tuple[int, int]:
        """
        Return image width and height.
        """

        from PIL import Image

        try:

            with Image.open(
                image_path
            ) as image:

                return image.size

        except Exception:

            return 0, 0

    @staticmethod
    def _yolo_to_xyxy(
        x_center: float,
        y_center: float,
        bbox_width: float,
        bbox_height: float,
        image_width: int,
        image_height: int,
    ) -> List[float]:
        """
        Convert normalized YOLO bbox to pixel xyxy.
        """

        x_center *= image_width
        y_center *= image_height

        bbox_width *= image_width
        bbox_height *= image_height

        x1 = (
            x_center
            - bbox_width / 2
        )

        y1 = (
            y_center
            - bbox_height / 2
        )

        x2 = (
            x_center
            + bbox_width / 2
        )

        y2 = (
            y_center
            + bbox_height / 2
        )

        return [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        ]

