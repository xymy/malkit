from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..typing import FilePath

__all__ = ["LabeledDataset", "LabeledImageDataset"]


def pil_loader(path: FilePath) -> Image.Image:
    with open(path, "rb") as f:
        image = Image.open(f)
        return image.convert()


class LabeledDataset(Dataset):
    def __init__(
        self, root: FilePath, loader: Callable[[FilePath], Any], labels: pd.DataFrame, *, cat: bool = False
    ) -> None:
        self.root = Path(root)
        self.loader = loader
        self.labels = labels
        self.cat = cat

        self._samples = [str(item) for item in labels.iloc[:, 0]]
        self._targets = [int(item) for item in labels.iloc[:, 1]]

    @property
    def samples(self):
        return self._samples

    @property
    def targets(self):
        return self._targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filename = self.samples[index]
        target = self.targets[index]

        if self.cat:
            sample_path = self.root / str(target) / filename
        else:
            sample_path = self.root / filename

        sample = self.loader(sample_path)
        return sample, target

    def __len__(self) -> int:
        return len(self.labels)


class LabeledImageDataset(LabeledDataset):
    def __init__(
        self,
        root: FilePath,
        labels: pd.DataFrame,
        *,
        cat: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, pil_loader, labels, cat=cat)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, target = super().__getitem__(index)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
