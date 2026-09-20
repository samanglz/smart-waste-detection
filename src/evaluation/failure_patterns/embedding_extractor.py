from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingExtractor(ABC):
    """
    Interface for extracting image embeddings.

    Implementations may use different embedding backends
    such as a CNN, CLIP, YOLO backbone, or another encoder.
    """

    @abstractmethod
    def extract(self, image_path: Path) -> np.ndarray:
        """
        Extract an embedding from one image.

        Args:
            image_path:
                Path to the image.

        Returns:
            1-D numpy array representing the image embedding.
        """
        raise NotImplementedError

