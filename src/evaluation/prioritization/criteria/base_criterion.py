from abc import ABC, abstractmethod
from typing import Any, Dict


class PriorityCriterion(ABC):
    """
    Define the contract for error-prioritization criteria.

    A criterion is responsible for extracting a comparable score
    from a structured evaluation item and defining how that score
    should be interpreted during ranking.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the criterion.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def lower_is_worse(self) -> bool:
        """
        Indicate whether lower scores represent worse performance.

        Returns:
            True:
                Lower scores receive higher priority.

            False:
                Higher scores receive higher priority.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, item: Dict[str, Any]) -> float:
        """
        Extract the criterion score from an evaluation item.

        Args:
            item:
                Structured evaluation or error data.

        Returns:
            Numeric score used for ranking.
        """
        raise NotImplementedError