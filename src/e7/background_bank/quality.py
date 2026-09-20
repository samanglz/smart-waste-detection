from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class QualityResult:
    """
    Quality analysis result for one background image.
    """

    sharpness_score: float
    brightness_score: float
    contrast_score: float
    overall_score: float

    is_blurry: bool
    is_too_dark: bool
    is_too_bright: bool
    is_low_contrast: bool


class BackgroundQualityAnalyzer:
    """
    Analyze visual quality of background images.

    Current checks:
        - sharpness / blur
        - brightness
        - contrast
    """

    def __init__(
        self,
        blur_threshold: float = 50.0,
        dark_threshold: float = 35.0,
        bright_threshold: float = 220.0,
        contrast_threshold: float = 25.0,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.contrast_threshold = contrast_threshold

    def analyze(self, image) -> QualityResult:
        """
        Analyze the quality of an OpenCV BGR image.
        """

        if image is None:
            raise ValueError("Image cannot be None.")

        if image.size == 0:
            raise ValueError("Image cannot be empty.")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        sharpness = self._calculate_sharpness(gray)
        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)

        sharpness_score = self._normalize_sharpness(sharpness)

        brightness_score = self._calculate_brightness_score(
            brightness
        )

        contrast_score = self._normalize_contrast(contrast)

        overall_score = (
            0.5 * sharpness_score
            + 0.25 * brightness_score
            + 0.25 * contrast_score
        )

        return QualityResult(
            sharpness_score=sharpness_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            overall_score=float(overall_score),
            is_blurry=sharpness < self.blur_threshold,
            is_too_dark=brightness < self.dark_threshold,
            is_too_bright=brightness > self.bright_threshold,
            is_low_contrast=contrast < self.contrast_threshold,
        )

    @staticmethod
    def _calculate_sharpness(gray) -> float:
        """
        Estimate image sharpness using Laplacian variance.
        """

        return float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

    @staticmethod
    def _calculate_brightness(gray) -> float:
        """
        Calculate mean grayscale brightness.
        """

        return float(gray.mean())

    @staticmethod
    def _calculate_contrast(gray) -> float:
        """
        Calculate contrast using grayscale standard deviation.
        """

        return float(gray.std())

    @staticmethod
    def _normalize_sharpness(
        sharpness: float,
    ) -> float:
        """
        Convert sharpness to a 0..1 score.
        """

        return float(
            min(
                1.0,
                sharpness / 500.0,
            )
        )

    @staticmethod
    def _calculate_brightness_score(
        brightness: float,
    ) -> float:
        """
        Give the highest score to approximately
        middle-range brightness.
        """

        distance_from_center = abs(
            brightness - 127.5
        )

        score = 1.0 - (
            distance_from_center / 127.5
        )

        return float(
            max(
                0.0,
                min(1.0, score),
            )
        )

    @staticmethod
    def _normalize_contrast(
        contrast: float,
    ) -> float:
        """
        Convert contrast to a 0..1 score.
        """

        return float(
            min(
                1.0,
                contrast / 64.0,
            )
        )