from pathlib import Path

from .base_dataset import BaseDataset


class COCODataset(BaseDataset):

    def __init__(self, dataset_root: Path):

        super().__init__(dataset_root)

    def load(self):

        return self.dataset_root

    def validate(self):

        return True