from __future__ import annotations

import cv2
import numpy as np

from src.e7.background_bank.quality import BackgroundQualityAnalyzer


def _create_textured_image(
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # ایجاد جزئیات و لبه‌های زیاد
    for y in range(0, height, 20):
        cv2.line(
            image,
            (0, y),
            (width, y),
            (255, 255, 255),
            2,
        )

    for x in range(0, width, 20):
        cv2.line(
            image,
            (x, 0),
            (x, height),
            (255, 255, 255),
            2,
        )

    return image


def test_quality_analyzer() -> None:
    analyzer = BackgroundQualityAnalyzer()

    image = _create_textured_image()

    result = analyzer.analyze(image)

    assert result is not None

    assert 0.0 <= result.sharpness_score <= 1.0
    assert 0.0 <= result.brightness_score <= 1.0
    assert 0.0 <= result.contrast_score <= 1.0
    assert 0.0 <= result.overall_score <= 1.0

    assert result.is_blurry is False


def test_quality_analyzer_rejects_empty_image() -> None:
    analyzer = BackgroundQualityAnalyzer()

    empty_image = np.empty(
        (0, 0, 3),
        dtype=np.uint8,
    )

    try:
        analyzer.analyze(empty_image)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_quality_analyzer_rejects_none() -> None:
    analyzer = BackgroundQualityAnalyzer()

    try:
        analyzer.analyze(None)
        assert False, "Expected ValueError"
    except ValueError:
        pass    