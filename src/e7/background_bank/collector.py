from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import cv2

from src.e7.background_bank.metadata import BackgroundMetadata


class BackgroundCollector:
    """
    Collect background images into the E7 Background Bank
    and generate initial metadata for each image.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    def __init__(
        self,
        output_dir: Path,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"

    def collect(
        self,
        source_dir: Path,
        category: str,
        source: str = "internet",
    ) -> list[BackgroundMetadata]:
        """
        Collect images from source_dir into the Background Bank.

        Images are copied into:
            output_dir/images/<category>/

        Returns:
            List of generated metadata objects.
        """
        source_dir = Path(source_dir)

        if not source_dir.exists():
            raise FileNotFoundError(
                f"Source directory does not exist: {source_dir}"
            )

        if not source_dir.is_dir():
            raise NotADirectoryError(
                f"Source path is not a directory: {source_dir}"
            )

        category = category.strip().lower()

        if not category:
            raise ValueError("Category cannot be empty.")

        category_dir = self.images_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        metadata_list: list[BackgroundMetadata] = []

        image_paths = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        for image_path in image_paths:
            metadata = self._collect_one(
                image_path=image_path,
                category=category,
                source=source,
                category_dir=category_dir,
            )

            if metadata is not None:
                metadata_list.append(metadata)

        return metadata_list

    def _collect_one(
        self,
        image_path: Path,
        category: str,
        source: str,
        category_dir: Path,
    ) -> BackgroundMetadata | None:
        """
        Copy one image and generate its initial metadata.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            return None

        height, width = image.shape[:2]

        if width <= 0 or height <= 0:
            return None

        background_id = self._generate_background_id(image_path)

        output_name = f"{background_id}{image_path.suffix.lower()}"
        output_path = category_dir / output_name

        shutil.copy2(image_path, output_path)

        return BackgroundMetadata(
            background_id=background_id,
            image_path=str(output_path),
            category=category,
            source=source,
            width=width,
            height=height,
            aspect_ratio=width / height,
            source_id=image_path.stem,
        )

    @staticmethod
    def _generate_background_id(image_path: Path) -> str:
        """
        Generate a deterministic ID from the source path.
        """

        normalized_path = str(image_path.resolve()).encode("utf-8")

        digest = hashlib.sha256(normalized_path).hexdigest()

        return f"bg_{digest[:12]}"