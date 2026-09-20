from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BackgroundManifest:
    """
    Read and write E7 Background Bank manifests.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Manifest does not exist: {self.path}"
            )

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "Background manifest must contain a list."
            )

        return data

    def save(
        self,
        entries: list[dict[str, Any]],
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                entries,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def count(self) -> int:
        return len(self.load())

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for entry in self.load():
            category = str(
                entry.get("category", "unknown")
            )

            counts[category] = (
                counts.get(category, 0) + 1
            )

        return counts