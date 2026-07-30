from abc import ABC, abstractmethod
from pathlib import Path


class BaseDataset(ABC):
    """
    Base interface for all dataset implementations.
    """

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)

    @abstractmethod
    def prepare(self) -> None:
        """Prepare dataset before training."""
        raise NotImplementedError

    @abstractmethod
    def get_train_path(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def get_val_path(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def get_test_path(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def get_num_classes(self) -> int:
        raise NotImplementedError