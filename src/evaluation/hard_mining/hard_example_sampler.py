from pathlib import Path
from typing import Dict, List, Any
import json

from src.logging.logger import get_logger


logger = get_logger(__name__)


class HardExampleSampler:
    """
    Build image-level sampling weights from hard training objects.

    Important:
        - Uses TRAIN samples only.
        - Does not copy or modify images.
        - Does not add TEST samples.
        - Hardness is derived from hard_samples_manifest.json.

    Since YOLO training operates on images, an image is considered
    a hard image if it contains at least one hard training object.
    """

    def __init__(
        self,
        manifest_path: Path,
        hard_weight: float = 3.0,
    ):
        """
        Args:
            manifest_path:
                Path to hard_samples_manifest.json.

            hard_weight:
                Sampling weight assigned to hard images.

                Example:
                    normal image = 1.0
                    hard image   = 3.0
        """

        self.manifest_path = Path(
            manifest_path
        )

        if hard_weight <= 0:
            raise ValueError(
                "hard_weight must be greater than 0."
            )

        self.hard_weight = float(
            hard_weight
        )

        self.manifest = self._load_manifest()

    # ============================================================
    # Public API
    # ============================================================

    def build_sampling_weights(
        self,
        train_images_dir: Path,
    ) -> Dict[str, float]:
        """
        Build image-level sampling weights.

        Returns:

            {
                "image1.jpg": 1.0,
                "image2.jpg": 3.0,
                ...
            }

        Normal images:
            weight = 1.0

        Hard images:
            weight = hard_weight
        """

        train_images_dir = Path(
            train_images_dir
        ).resolve()

        hard_images = self._collect_hard_images()

        weights = {}

        for image_path in self._collect_train_images(
            train_images_dir
        ):

            resolved_path = image_path.resolve()

            if resolved_path in hard_images:
                weights[str(resolved_path)] = (
                    self.hard_weight
                )
            else:
                weights[str(resolved_path)] = 1.0

        logger.info(
            "TRAIN images: %d",
            len(weights),
        )

        logger.info(
            "Hard TRAIN images: %d",
            len(hard_images),
        )

        logger.info(
            "Normal TRAIN images: %d",
            sum(
                1
                for weight in weights.values()
                if weight == 1.0
            ),
        )

        return weights

    def get_hard_images(self) -> List[Path]:
        """
        Return unique hard training images.
        """

        return sorted(
            self._collect_hard_images()
        )

    def get_statistics(
        self,
        train_images_dir: Path,
    ) -> Dict[str, Any]:
        """
        Return sampling statistics.
        """

        weights = self.build_sampling_weights(
            train_images_dir
        )

        hard_count = sum(
            1
            for weight in weights.values()
            if weight > 1.0
        )

        normal_count = (
            len(weights) - hard_count
        )

        return {
            "total_train_images": len(
                weights
            ),
            "hard_images": hard_count,
            "normal_images": normal_count,
            "hard_weight": self.hard_weight,
        }

    # ============================================================
    # Manifest
    # ============================================================

    def _load_manifest(self) -> Dict[str, Any]:
        """
        Load hard samples manifest.
        """

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
        ) as f:

            manifest = json.load(f)

        if "hard_samples" not in manifest:
            raise ValueError(
                "Manifest does not contain "
                "'hard_samples'."
            )

        return manifest

    # ============================================================
    # Hard image extraction
    # ============================================================

    def _collect_hard_images(self) -> set[Path]:
        """
        Extract unique TRAIN image paths from manifest.
        """

        hard_images = set()

        for sample in self.manifest[
            "hard_samples"
        ]:

            image_path = Path(
                sample["image_path"]
            ).resolve()

            hard_images.add(
                image_path
            )

        return hard_images

    # ============================================================
    # TRAIN image discovery
    # ============================================================

    @staticmethod
    def _collect_train_images(
        train_images_dir: Path,
    ) -> List[Path]:
        """
        Collect all training images.
        """

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        images = []

        for path in train_images_dir.iterdir():

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            images.append(path)

        return images