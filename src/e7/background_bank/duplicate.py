from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DuplicateResult:
    """
    Duplicate analysis result for one image.
    """

    image_path: str

    content_hash: str | None
    perceptual_hash: str | None

    readable: bool


class BackgroundDuplicateDetector:
    """
    Detect exact and near duplicates between background images.

    Exact duplicates:
        SHA-256 content hash

    Near duplicates:
        Perceptual hash (pHash-like DCT representation)
    """

    def __init__(
        self,
        hash_size: int = 8,
        dct_size: int = 32,
    ) -> None:
        if hash_size <= 0:
            raise ValueError("hash_size must be greater than zero.")

        if dct_size <= 0:
            raise ValueError("dct_size must be greater than zero.")

        if dct_size < hash_size:
            raise ValueError(
                "dct_size must be greater than or equal to hash_size."
            )

        self.hash_size = hash_size
        self.dct_size = dct_size

    def analyze(
        self,
        image_path: Path,
    ) -> DuplicateResult:
        """
        Calculate exact and perceptual hashes for one image.
        """

        image_path = Path(image_path)

        if not image_path.exists() or not image_path.is_file():
            return DuplicateResult(
                image_path=str(image_path),
                content_hash=None,
                perceptual_hash=None,
                readable=False,
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            return DuplicateResult(
                image_path=str(image_path),
                content_hash=None,
                perceptual_hash=None,
                readable=False,
            )

        content_hash = self._calculate_content_hash(
            image_path
        )

        perceptual_hash = self._calculate_perceptual_hash(
            image
        )

        return DuplicateResult(
            image_path=str(image_path),
            content_hash=content_hash,
            perceptual_hash=perceptual_hash,
            readable=True,
        )

    @staticmethod
    def _calculate_content_hash(
        image_path: Path,
    ) -> str:
        """
        Calculate SHA-256 from the actual file bytes.

        This detects exact file duplicates.
        """

        sha256 = hashlib.sha256()

        with image_path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _calculate_perceptual_hash(
        self,
        image: np.ndarray,
    ) -> str:
        """
        Calculate a DCT-based perceptual hash.

        Images with visually similar structure tend
        to produce similar hashes even after resizing
        or mild compression.
        """

        resized = cv2.resize(
            image,
            (self.dct_size, self.dct_size),
            interpolation=cv2.INTER_AREA,
        )

        resized = np.float32(resized)

        dct = cv2.dct(resized)

        low_frequency = dct[
            : self.hash_size,
            : self.hash_size,
        ]

        median = np.median(low_frequency)

        bits = low_frequency > median

        return "".join(
            "1" if bit else "0"
            for bit in bits.flatten()
        )

    @staticmethod
    def hamming_distance(
        hash_a: str,
        hash_b: str,
    ) -> int:
        """
        Calculate Hamming distance between two
        perceptual hashes.
        """

        if len(hash_a) != len(hash_b):
            raise ValueError(
                "Hashes must have the same length."
            )

        return sum(
            bit_a != bit_b
            for bit_a, bit_b in zip(
                hash_a,
                hash_b,
            )
        )

    def are_near_duplicates(
        self,
        hash_a: str,
        hash_b: str,
        max_distance: int = 8,
    ) -> bool:
        """
        Determine whether two perceptual hashes
        represent near-duplicate images.
        """

        if max_distance < 0:
            raise ValueError(
                "max_distance cannot be negative."
            )

        distance = self.hamming_distance(
            hash_a,
            hash_b,
        )

        return distance <= max_distance