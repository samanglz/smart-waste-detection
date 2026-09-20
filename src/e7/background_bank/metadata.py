from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BackgroundMetadata:
    """
    Metadata describing one background image in the E7 Background Bank.
    """

    background_id: str
    image_path: str

    category: str
    source: str

    width: int
    height: int
    aspect_ratio: float

    contains_waste: bool = False
    contains_people: bool = False

    quality_score: float | None = None
    usable: bool = True

    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to a JSON-serializable dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackgroundMetadata":
        """
        Create metadata from a dictionary.
        """
        return cls(
            background_id=str(data["background_id"]),
            image_path=str(data["image_path"]),
            category=str(data["category"]),
            source=str(data["source"]),
            width=int(data["width"]),
            height=int(data["height"]),
            aspect_ratio=float(data["aspect_ratio"]),
            contains_waste=bool(data.get("contains_waste", False)),
            contains_people=bool(data.get("contains_people", False)),
            quality_score=(
                float(data["quality_score"])
                if data.get("quality_score") is not None
                else None
            ),
            usable=bool(data.get("usable", True)),
            source_id=(
                str(data["source_id"])
                if data.get("source_id") is not None
                else None
            ),
        )