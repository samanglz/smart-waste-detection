"""
Generation count strategies for E7 Dataset Builder.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationDecision:
    category: str
    count: int
    area_ratio: float


class GenerationCountStrategy(ABC):
    """
    Base interface for deciding how many synthetic samples
    should be generated from one object.
    """

    @abstractmethod
    def decide(
        self,
        object_area_pixels: int,
        image_width: int,
        image_height: int,
    ) -> GenerationDecision:
        raise NotImplementedError


class AreaBasedGenerationStrategy(GenerationCountStrategy):
    """
    Decide generation count based on object area ratio.

    Small  -> 3 generations
    Medium -> 2 generations
    Large  -> 1 generation
    """

    def __init__(
        self,
        small_threshold: float = 0.05,
        medium_threshold: float = 0.20,
    ):
        if not 0.0 < small_threshold < medium_threshold < 1.0:
            raise ValueError(
                "Thresholds must satisfy: "
                "0 < small < medium < 1."
            )

        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold

    def decide(
        self,
        object_area_pixels: int,
        image_width: int,
        image_height: int,
    ) -> GenerationDecision:

        if object_area_pixels <= 0:
            raise ValueError("Object area must be positive.")

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "Image dimensions must be positive."
            )

        image_area = image_width * image_height
        area_ratio = object_area_pixels / image_area

        if area_ratio < self.small_threshold:
            category = "small"
            count = 3

        elif area_ratio < self.medium_threshold:
            category = "medium"
            count = 2

        else:
            category = "large"
            count = 1

        return GenerationDecision(
            category=category,
            count=count,
            area_ratio=area_ratio,
        )