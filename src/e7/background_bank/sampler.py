from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


class BackgroundSampler:
    """
    Sample backgrounds from the E7 Background Bank.

    By default, sampling is proportional to the number of
    approved images in each category, so the sampler does not
    impose an artificial category distribution.
    """

    def __init__(
        self,
        backgrounds: list[dict[str, Any]],
        seed: int = 42,
    ) -> None:
        if not backgrounds:
            raise ValueError(
                "Background pool cannot be empty."
            )

        self._rng = random.Random(seed)

        self._backgrounds = [
            dict(background)
            for background in backgrounds
        ]

        self._categories: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)

        for background in self._backgrounds:
            category = str(
                background.get("category", "unknown")
            )

            self._categories[category].append(
                background
            )

        self._remaining = {
            category: list(items)
            for category, items
            in self._categories.items()
        }

        for items in self._remaining.values():
            self._rng.shuffle(items)

    def sample(self) -> dict[str, Any]:
        """
        Sample one background.

        Categories are selected according to the number
        of remaining backgrounds in each category.
        """

        available_categories = [
            category
            for category, items
            in self._remaining.items()
            if items
        ]

        if not available_categories:
            self._reset_cycle()

            available_categories = [
                category
                for category, items
                in self._remaining.items()
                if items
            ]

        weights = [
            len(self._remaining[category])
            for category in available_categories
        ]

        category = self._rng.choices(
            available_categories,
            weights=weights,
            k=1,
        )[0]

        return self._remaining[category].pop()

    def remaining(self) -> int:
        """
        Return number of backgrounds remaining
        in the current cycle.
        """

        return sum(
            len(items)
            for items in self._remaining.values()
        )

    def categories(self) -> dict[str, int]:
        """
        Return the number of backgrounds in each category.
        """

        return {
            category: len(items)
            for category, items
            in self._categories.items()
        }

    def _reset_cycle(self) -> None:
        """
        Start a new sampling cycle.
        """

        self._remaining = {
            category: list(items)
            for category, items
            in self._categories.items()
        }

        for items in self._remaining.values():
            self._rng.shuffle(items)