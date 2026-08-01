from abc import ABC, abstractmethod
from pathlib import Path


class BaseDataset(ABC):

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)

    @abstractmethod
    def load(self):
        """Load dataset resources."""
        pass

    @abstractmethod
    def validate(self):
        """Validate dataset integrity."""
        pass

    @abstractmethod
    def get_metadata(self):
        """Return dataset information."""
        pass