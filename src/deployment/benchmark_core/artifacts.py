from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib


@dataclass(frozen=True)
class ArtifactInfo:
    """
    Immutable metadata describing a benchmarked model artifact.

    The SHA256 hash provides a reproducible identity for the exact
    model file used during benchmarking.
    """

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def calculate_sha256(
    file_path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate the SHA256 hash of a file using streaming I/O.

    Parameters
    ----------
    file_path:
        Path to the file.

    chunk_size:
        Number of bytes read per iteration.

    Returns
    -------
    str
        Lowercase hexadecimal SHA256 digest.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Artifact path is not a file: {path}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def get_artifact_info(
    file_path: str | Path,
) -> ArtifactInfo:
    """
    Collect reproducibility metadata for a model artifact.

    Parameters
    ----------
    file_path:
        Path to the model artifact.

    Returns
    -------
    ArtifactInfo
        Path, SHA256 hash, and exact file size.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Artifact path is not a file: {path}"
        )

    return ArtifactInfo(
        path=str(path.resolve()),
        sha256=calculate_sha256(path),
        size_bytes=path.stat().st_size,
    )