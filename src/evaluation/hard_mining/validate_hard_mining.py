from pathlib import Path
from typing import Dict, Any, List
import json

from src.logging.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# Configuration
# ============================================================

MINING_RESULT_PATH = Path(
    "outputs/hard_example_mining/E2/hard_example_mining.json"
)

TRAIN_IMAGES_DIR = Path(
    "data/processed/train/images"
).resolve()

OUTPUT_PATH = Path(
    "outputs/hard_example_mining/E2/hard_samples_manifest.json"
).resolve()


# ============================================================
# Load
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Validation
# ============================================================

def is_train_image(image_path: Path) -> bool:
    """
    Verify that image belongs to TRAIN images directory.
    """

    try:
        image_path.resolve().relative_to(
            TRAIN_IMAGES_DIR
        )
        return True

    except ValueError:
        return False


def validate_sample(
    sample: Dict[str, Any],
) -> List[str]:

    errors = []

    required_keys = {
        "image_path",
        "bbox",
        "class_name",
        "similarity",
        "matched_test_cases",
    }

    missing = required_keys - sample.keys()

    if missing:
        errors.append(
            f"missing_keys={sorted(missing)}"
        )
        return errors

    # --------------------------------------------------------
    # Image path
    # --------------------------------------------------------

    image_path = Path(
        sample["image_path"]
    ).resolve()

    if not is_train_image(image_path):
        errors.append(
            f"not_train_sample={image_path}"
        )

    elif not image_path.exists():
        errors.append(
            f"image_not_found={image_path}"
        )

    # --------------------------------------------------------
    # BBox
    # --------------------------------------------------------

    bbox = sample["bbox"]

    if not isinstance(bbox, list):
        errors.append("bbox_not_list")

    elif len(bbox) != 4:
        errors.append(
            f"invalid_bbox_length={len(bbox)}"
        )

    else:
        try:
            x1, y1, x2, y2 = map(float, bbox)

            if x2 <= x1 or y2 <= y1:
                errors.append(
                    f"invalid_bbox={bbox}"
                )

        except (TypeError, ValueError):
            errors.append(
                f"invalid_bbox_values={bbox}"
            )

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    try:
        similarity = float(
            sample["similarity"]
        )

        if not -1.0 <= similarity <= 1.0:
            errors.append(
                f"invalid_similarity={similarity}"
            )

    except (TypeError, ValueError):
        errors.append("invalid_similarity")

    # --------------------------------------------------------
    # Matched test cases
    # --------------------------------------------------------

    try:
        matched = int(
            sample["matched_test_cases"]
        )

        if matched < 1:
            errors.append(
                f"invalid_matched_test_cases={matched}"
            )

    except (TypeError, ValueError):
        errors.append(
            "invalid_matched_test_cases"
        )

    return errors


# ============================================================
# Deduplication
# ============================================================

def sample_key(
    sample: Dict[str, Any],
) -> str:

    image_path = str(
        Path(sample["image_path"]).resolve()
    )

    bbox = sample["bbox"]

    bbox_key = ",".join(
        f"{float(value):.4f}"
        for value in bbox
    )

    return f"{image_path}|{bbox_key}"


# ============================================================
# Main validation
# ============================================================

def validate_mining_result(
    mining_result: Dict[str, Any],
) -> Dict[str, Any]:

    confusions = mining_result.get(
        "confusions",
        [],
    )

    valid_samples = []
    invalid_samples = []

    seen = set()

    total_samples = 0
    duplicate_samples = 0

    # --------------------------------------------------------
    # Iterate over confusions
    # --------------------------------------------------------

    for confusion in confusions:

        rank = confusion.get(
            "rank"
        )

        gt_class = confusion.get(
            "gt_class"
        )

        pred_class = confusion.get(
            "pred_class"
        )

        train_samples = confusion.get(
            "train_samples",
            [],
        )

        logger.info(
            "Validating #%s: %s → %s | samples=%d",
            rank,
            gt_class,
            pred_class,
            len(train_samples),
        )

        for sample in train_samples:

            total_samples += 1

            errors = validate_sample(
                sample
            )

            if errors:

                invalid_samples.append(
                    {
                        "rank": rank,
                        "gt_class": gt_class,
                        "pred_class": pred_class,
                        "sample": sample,
                        "errors": errors,
                    }
                )

                continue

            # ------------------------------------------------
            # Deduplicate
            # ------------------------------------------------

            key = sample_key(sample)

            if key in seen:
                duplicate_samples += 1
                continue

            seen.add(key)

            # ------------------------------------------------
            # Create clean manifest record
            # ------------------------------------------------

            valid_samples.append(
                {
                    "image_path": str(
                        Path(
                            sample["image_path"]
                        ).resolve()
                    ),
                    "bbox": sample["bbox"],
                    "class_name": sample[
                        "class_name"
                    ],
                    "similarity": float(
                        sample["similarity"]
                    ),
                    "matched_test_cases": int(
                        sample[
                            "matched_test_cases"
                        ]
                    ),
                    "source_confusion": {
                        "gt_class": gt_class,
                        "pred_class": pred_class,
                        "rank": rank,
                    },
                }
            )

    # --------------------------------------------------------
    # Rank final samples
    # --------------------------------------------------------

    valid_samples.sort(
        key=lambda x: (
            x["similarity"],
            x["matched_test_cases"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "source": {
            "mining_result": str(
                MINING_RESULT_PATH
            ),
            "train_images_dir": str(
                TRAIN_IMAGES_DIR
            ),
            "test_samples_used_as_training": False,
        },
        "statistics": {
            "total_samples_in_mining_result": (
                total_samples
            ),
            "valid_samples": len(
                valid_samples
            ),
            "invalid_samples": len(
                invalid_samples
            ),
            "duplicate_samples": (
                duplicate_samples
            ),
        },
        "hard_samples": valid_samples,
        "invalid_samples": invalid_samples,
    }


# ============================================================
# Save
# ============================================================

def save_manifest(
    manifest: Dict[str, Any],
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Entry point
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info(
        "VALIDATING HARD EXAMPLE MINING"
    )
    logger.info("=" * 70)

    logger.info(
        "Mining result: %s",
        MINING_RESULT_PATH,
    )

    mining_result = load_json(
        MINING_RESULT_PATH
    )

    manifest = validate_mining_result(
        mining_result
    )

    save_manifest(
        manifest
    )

    stats = manifest[
        "statistics"
    ]

    logger.info("=" * 70)
    logger.info(
        "VALIDATION COMPLETED"
    )
    logger.info("=" * 70)

    logger.info(
        "Total mining samples: %d",
        stats[
            "total_samples_in_mining_result"
        ],
    )

    logger.info(
        "Valid samples: %d",
        stats[
            "valid_samples"
        ],
    )

    logger.info(
        "Invalid samples: %d",
        stats[
            "invalid_samples"
        ],
    )

    logger.info(
        "Duplicate samples: %d",
        stats[
            "duplicate_samples"
        ],
    )

    logger.info(
        "Manifest: %s",
        OUTPUT_PATH,
    )

    if stats["invalid_samples"] > 0:

        logger.warning(
            "Some samples were rejected. "
            "Check invalid_samples in the manifest."
        )

    else:

        logger.info(
            "All selected samples belong to TRAIN."
        )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()