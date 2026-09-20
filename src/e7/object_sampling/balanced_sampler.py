"""
Balanced object sampler for E7.

The sampler ensures that objects are selected without replacement
until the entire Object Bank has been consumed.

After that, a new shuffled cycle begins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_sampler import ObjectSampler


class BalancedObjectSampler(ObjectSampler):
    """
    Select objects from the Object Bank in balanced cycles.

    Each object is selected at most once per cycle.

    Example
    -------
    If the Object Bank contains 1677 objects and we need 1673
    selections, 1673 different objects will be selected.

    If we need more than 1677 selections, a new shuffled cycle
    starts automatically.
    """

    def __init__(
        self,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)

        self._shuffled_pool: list[dict[str, Any]] = []
        self._cursor = 0

    def sample(
        self,
        object_pool: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not object_pool:
            raise ValueError(
                "Object pool cannot be empty."
            )

        self._ensure_cycle(object_pool)

        object_entry = self._shuffled_pool[self._cursor]

        self._cursor += 1

        return object_entry

    def _ensure_cycle(
        self,
        object_pool: list[dict[str, Any]],
    ) -> None:

        if (
            not self._shuffled_pool
            or self._cursor >= len(self._shuffled_pool)
        ):
            self._shuffled_pool = list(object_pool)

            self.rng.shuffle(
                self._shuffled_pool
            )

            self._cursor = 0