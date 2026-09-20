"""
Base interface for E7 object sampling strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ObjectSampler(ABC):
    """
    Base interface for selecting objects from the E7 Object Bank.
    """

    @abstractmethod
    def sample(
        self,
        object_pool: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Select one object from the object pool.

        Parameters
        ----------
        object_pool:
            Available objects from the E7 Object Bank.

        Returns
        -------
        dict
            Selected object entry.
        """
        raise NotImplementedError