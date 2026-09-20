from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from src.e7.background_bank.duplicate import (
    BackgroundDuplicateDetector,
)
from src.e7.background_bank.quality import (
    BackgroundQualityAnalyzer,
)


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of validating one background image.
    """

    image_path: str

    width: int
    height: int
    aspect_ratio: float

    quality_score: float | None

    content_hash: str | None
    perceptual_hash: str | None

    usable: bool

    reasons: tuple[str, ...]


class BackgroundValidator:
    """
    Validate one background image.

    Current checks:
        - file existence
        - image readability
        - minimum resolution
        - aspect ratio
        - visual quality
        - exact duplicate
        - near duplicate
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    def __init__(
        self,
        min_width: int = 640,
        min_height: int = 480,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.5,
        quality_analyzer: BackgroundQualityAnalyzer | None = None,
        duplicate_detector: BackgroundDuplicateDetector | None = None,
        min_quality_score: float = 0.20,
        near_duplicate_distance: int = 8,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height

        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.quality_analyzer = (
            quality_analyzer
            if quality_analyzer is not None
            else BackgroundQualityAnalyzer()
        )

        self.duplicate_detector = (
            duplicate_detector
            if duplicate_detector is not None
            else BackgroundDuplicateDetector()
        )

        self.min_quality_score = min_quality_score
        self.near_duplicate_distance = near_duplicate_distance

    def validate(
        self,
        image_path: Path,
        existing_content_hashes: set[str] | None = None,
        existing_perceptual_hashes: list[str] | None = None,
    ) -> ValidationResult:
        """
        Validate one background image.

        existing_content_hashes:
            SHA-256 hashes of already accepted images.

        existing_perceptual_hashes:
            Perceptual hashes of already accepted images.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            return self._invalid_result(
                image_path=image_path,
                reason="file_not_found",
            )

        if not image_path.is_file():
            return self._invalid_result(
                image_path=image_path,
                reason="not_a_file",
            )

        if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return self._invalid_result(
                image_path=image_path,
                reason="unsupported_extension",
            )

        image = cv2.imread(str(image_path))

        if image is None:
            return self._invalid_result(
                image_path=image_path,
                reason="image_unreadable",
            )

        height, width = image.shape[:2]

        if width <= 0 or height <= 0:
            return ValidationResult(
                image_path=str(image_path),
                width=width,
                height=height,
                aspect_ratio=0.0,
                quality_score=None,
                content_hash=None,
                perceptual_hash=None,
                usable=False,
                reasons=("invalid_dimensions",),
            )

        aspect_ratio = width / height

        reasons: list[str] = []

        if width < self.min_width:
            reasons.append("width_too_small")

        if height < self.min_height:
            reasons.append("height_too_small")

        if aspect_ratio < self.min_aspect_ratio:
            reasons.append("aspect_ratio_too_small")

        if aspect_ratio > self.max_aspect_ratio:
            reasons.append("aspect_ratio_too_large")

        # ---------------------------------------------------------
        # Quality analysis
        # ---------------------------------------------------------

        quality_result = self.quality_analyzer.analyze(image)

        quality_score = quality_result.overall_score

        if quality_result.is_blurry:
            reasons.append("image_blurry")

        if quality_result.is_too_dark:
            reasons.append("image_too_dark")

        if quality_result.is_too_bright:
            reasons.append("image_too_bright")

        if quality_result.is_low_contrast:
            reasons.append("low_contrast")

        if quality_score < self.min_quality_score:
            reasons.append("quality_score_too_low")

        # ---------------------------------------------------------
        # Duplicate analysis
        # ---------------------------------------------------------

        duplicate_result = self.duplicate_detector.analyze(
            image_path
        )

        content_hash = duplicate_result.content_hash
        perceptual_hash = duplicate_result.perceptual_hash

        if (
            content_hash is not None
            and existing_content_hashes is not None
            and content_hash in existing_content_hashes
        ):
            reasons.append("exact_duplicate")

        if (
            perceptual_hash is not None
            and existing_perceptual_hashes
        ):
            for existing_hash in existing_perceptual_hashes:
                if self.duplicate_detector.are_near_duplicates(
                    perceptual_hash,
                    existing_hash,
                    max_distance=self.near_duplicate_distance,
                ):
                    reasons.append("near_duplicate")
                    break

        usable = len(reasons) == 0

        return ValidationResult(
            image_path=str(image_path),
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            quality_score=quality_score,
            content_hash=content_hash,
            perceptual_hash=perceptual_hash,
            usable=usable,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _invalid_result(
        image_path: Path,
        reason: str,
    ) -> ValidationResult:
        """
        Create a standard invalid validation result.
        """

        return ValidationResult(
            image_path=str(image_path),
            width=0,
            height=0,
            aspect_ratio=0.0,
            quality_score=None,
            content_hash=None,
            perceptual_hash=None,
            usable=False,
            reasons=(reason,),
        )