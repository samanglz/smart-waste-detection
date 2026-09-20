from pathlib import Path
from typing import Dict, List, Any

from src.logging.logger import get_logger

from src.evaluation.hard_mining.hard_case_similarity import (
    HardCaseSimilarityAnalyzer,
)

logger = get_logger(__name__)


class HardExampleMiner:
    """
    Mine hard TRAIN samples using prioritized TEST errors.

    Pipeline:

        priority_summary.json
                ↓
        prioritized confusions
                ↓
        TEST error cases
                ↓
        HardCaseSimilarityAnalyzer
                ↓
        TRAIN embeddings
                ↓
        similar TRAIN objects
                ↓
        hard TRAIN samples

    Important:
        TEST samples are used only as queries.

        No TEST image is ever selected as a TRAIN sample.
    """

    def __init__(
        self,
        similarity_analyzer: HardCaseSimilarityAnalyzer,
    ):
        self.analyzer = similarity_analyzer

    # ============================================================
    # Public API
    # ============================================================

    def mine_confusion(
        self,
        gt_class: str,
        pred_class: str,
        cases: List[Dict[str, Any]],
        top_k_per_case: int = 5,
        max_samples: int = 50,
    ) -> Dict[str, Any]:
        """
        Mine hard TRAIN samples for one prioritized confusion.

        Example:

            metal → glass

        TEST:
            metal objects incorrectly predicted as glass

        TRAIN mining:
            Find visually similar TRAIN metal objects.

        Args:
            gt_class:
                Ground-truth class.

            pred_class:
                Predicted class.

            cases:
                TEST error cases belonging to this confusion.

            top_k_per_case:
                Number of nearest TRAIN samples retrieved
                for each TEST case.

            max_samples:
                Maximum number of unique TRAIN samples
                returned for this confusion.
        """

        logger.info(
            "Mining hard TRAIN samples for %s → %s",
            gt_class,
            pred_class,
        )

        candidates: Dict[str, Dict[str, Any]] = {}

        # --------------------------------------------------------
        # Analyze each TEST error
        # --------------------------------------------------------

        for index, case in enumerate(cases):

            logger.info(
                "Mining query %d/%d",
                index + 1,
                len(cases),
            )

            # ----------------------------------------------------
            # Validate TEST case
            # ----------------------------------------------------

            self._validate_test_case(
                case=case,
                gt_class=gt_class,
                pred_class=pred_class,
            )

            # ----------------------------------------------------
            # TEST case → TRAIN embedding search
            # ----------------------------------------------------

            analysis = self.analyzer.analyze_case(
                case=case,
                top_k=top_k_per_case,
            )

            # ----------------------------------------------------
            # IMPORTANT
            #
            # For:
            #
            #     metal → glass
            #
            # we mine TRAIN samples belonging to:
            #
            #     metal
            #
            # NOT glass.
            # ----------------------------------------------------

            train_neighbors = (
                analysis
                .get("nearest_neighbors", {})
                .get(gt_class, [])
            )

            # ----------------------------------------------------
            # Collect TRAIN candidates
            # ----------------------------------------------------

            for neighbor in train_neighbors:

                train_image = Path(
                    neighbor["image_path"]
                ).resolve()

                similarity = float(
                    neighbor["similarity"]
                )

                bbox = neighbor["bbox"]

                # ------------------------------------------------
                # Safety check:
                # selected image MUST belong to TRAIN
                # ------------------------------------------------

                if not self._is_train_sample(
                    train_image
                ):
                    logger.error(
                        "Rejected non-train sample: %s",
                        train_image,
                    )
                    continue

                # ------------------------------------------------
                # Unique object identifier
                #
                # Same image may contain multiple objects.
                # Therefore:
                #
                # image + bbox
                #
                # is used as the unique key.
                # ------------------------------------------------

                key = self._sample_key(
                    train_image,
                    bbox,
                )

                if key not in candidates:

                    candidates[key] = {
                        "image_path": str(
                            train_image
                        ),
                        "bbox": bbox,
                        "class_name": gt_class,
                        "similarity": similarity,
                        "matched_test_cases": 1,
                    }

                else:

                    # Keep strongest similarity
                    candidates[key][
                        "similarity"
                    ] = max(
                        candidates[key]["similarity"],
                        similarity,
                    )

                    # Count how many TEST hard cases
                    # retrieved this TRAIN object.
                    candidates[key][
                        "matched_test_cases"
                    ] += 1

        # --------------------------------------------------------
        # Rank TRAIN candidates
        # --------------------------------------------------------

        ranked_samples = sorted(
            candidates.values(),
            key=lambda x: (
                x["similarity"],
                x["matched_test_cases"],
            ),
            reverse=True,
        )

        # --------------------------------------------------------
        # Limit number of samples
        # --------------------------------------------------------

        ranked_samples = ranked_samples[
            :max_samples
        ]

        # --------------------------------------------------------
        # Final result
        # --------------------------------------------------------

        result = {
            "gt_class": gt_class,
            "pred_class": pred_class,
            "total_test_cases": len(cases),
            "unique_train_candidates": len(
                candidates
            ),
            "selected_train_samples": len(
                ranked_samples
            ),
            "train_samples": ranked_samples,
        }

        logger.info(
            "Selected %d TRAIN samples for %s → %s",
            len(ranked_samples),
            gt_class,
            pred_class,
        )

        return result

    # ============================================================
    # Prioritized Confusions
    # ============================================================

    def mine_prioritized_confusions(
        self,
        prioritized_confusions: List[
            Dict[str, Any]
        ],
        top_k_per_case: int = 5,
        max_samples_per_confusion: int = 50,
    ) -> Dict[str, Any]:
        """
        Mine TRAIN samples for all prioritized confusions.

        Priority information is expected to come from:

            ErrorPrioritizer.rank_confusions()

        or from the persisted:

            priority_summary.json

        This method does NOT calculate priority itself.
        """

        results = []

        for rank, confusion in enumerate(
            prioritized_confusions,
            start=1,
        ):

            gt_class = confusion[
                "gt_class"
            ]

            pred_class = confusion[
                "pred_class"
            ]

            cases = confusion[
                "cases"
            ]

            logger.info("=" * 70)

            logger.info(
                "Mining Priority #%d: %s → %s",
                rank,
                gt_class,
                pred_class,
            )

            logger.info("=" * 70)

            result = self.mine_confusion(
                gt_class=gt_class,
                pred_class=pred_class,
                cases=cases,
                top_k_per_case=top_k_per_case,
                max_samples=max_samples_per_confusion,
            )

            result["rank"] = rank

            results.append(result)

        return {
            "source": {
                "priority_source": (
                    "ErrorPrioritizer.rank_confusions"
                ),
                "test_as_query": True,
                "train_as_reference": True,
                "test_samples_added_to_train": False,
            },
            "confusions": results,
        }

    # ============================================================
    # Safety
    # ============================================================

    def _validate_test_case(
        self,
        case: Dict[str, Any],
        gt_class: str,
        pred_class: str,
    ) -> None:
        """
        Validate that the case belongs to
        the expected prioritized confusion.
        """

        if case["gt_class"] != gt_class:

            raise ValueError(
                "GT class mismatch: "
                f"expected={gt_class}, "
                f"actual={case['gt_class']}"
            )

        if case["pred_class"] != pred_class:

            raise ValueError(
                "Predicted class mismatch: "
                f"expected={pred_class}, "
                f"actual={case['pred_class']}"
            )

        required_keys = {
            "image_path",
            "gt_class",
            "pred_class",
            "gt_bbox",
            "pred_bbox",
            "iou",
            "confidence",
        }

        missing = (
            required_keys
            - case.keys()
        )

        if missing:

            raise ValueError(
                f"Missing keys in error case: "
                f"{missing}"
            )

    # ============================================================
    # TRAIN Safety
    # ============================================================

    def _is_train_sample(
        self,
        image_path: Path,
    ) -> bool:
        """
        Verify that an image belongs to
        the configured TRAIN directory.
        """

        train_dir = (
            self.analyzer
            .train_images_dir
            .resolve()
        )

        try:

            image_path.relative_to(
                train_dir
            )

            return True

        except ValueError:

            return False

    # ============================================================
    # Unique Sample Key
    # ============================================================

    @staticmethod
    def _sample_key(
        image_path: Path,
        bbox: List[float],
    ) -> str:
        """
        Generate a unique identifier for
        a TRAIN object.

        Same image can contain multiple
        objects, so bbox is included.
        """

        bbox_key = ",".join(
            f"{value:.4f}"
            for value in bbox
        )

        return (
            f"{image_path}|{bbox_key}"
        )