from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .._typing import FilePath

__all__ = ["LabeledDataset", "LabeledImageDataset", "UnlabeledDataset", "UnlabeledImageDataset"]


class PILLoader:
    def __init__(self, mode: Optional[str] = None) -> None:
        self.mode = mode

    def __call__(self, path: FilePath) -> Image.Image:
        with open(path, "rb") as f:
            image = Image.open(f)
            return image.convert(self.mode)


class LabeledDataset(Dataset):
    def __init__(
        self,
        root: FilePath,
        loader: Callable[[FilePath], Any],
        labels: pd.DataFrame,
        *,
        cat: bool = False,
        suffix: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.loader = loader
        self.labels = labels
        self.cat = cat
        self.suffix = suffix

        self._sample_names = [str(item) for item in labels.iloc[:, 0]]
        self._target_names = [str(item) for item in labels.iloc[:, 1]]

        self._index_to_class = sorted(set(self._target_names))
        self._class_to_index = {c: i for i, c in enumerate(self._index_to_class)}

    @property
    def sample_names(self) -> List[str]:
        return self._sample_names

    @property
    def target_names(self) -> List[str]:
        return self._target_names

    @property
    def index_to_class(self) -> List[str]:
        return self._index_to_class

    @property
    def class_to_index(self) -> Dict[str, int]:
        return self._class_to_index

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        sample_name = self._sample_names[index]
        target_name = self._target_names[index]

        if self.cat:
            sample_path = self.root / target_name / sample_name
        else:
            sample_path = self.root / sample_name

        # If ``suffix`` is an empty string, the suffix will be removed.
        if self.suffix is not None:
            sample_path = sample_path.with_suffix(self.suffix)

        sample = self.loader(sample_path)
        target = self._class_to_index[target_name]
        return sample, target

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        s = ""
        s += f"{type(self).__name__}:\n"
        s += f"    Root directory: {self.root}\n"
        s += f"    Number of samples: {len(self)}\n"
        s += f"    Number of classes: {len(self.index_to_class)}\n"
        return s


class LabeledImageDataset(LabeledDataset):
    def __init__(
        self,
        root: FilePath,
        labels: pd.DataFrame,
        *,
        cat: bool = False,
        suffix: Optional[str] = ".png",
        loader_mode: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        loader = PILLoader(loader_mode)
        super().__init__(root, loader, labels, cat=cat, suffix=suffix)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        sample, target = super().__getitem__(index)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class UnlabeledDataset(Dataset):
    def __init__(self, root: FilePath, loader: Callable[[FilePath], Any]) -> None:
        self.root = Path(root)
        self.loader = loader

        self._sample_paths = [entry for entry in self.root.iterdir()]

    def __getitem__(self, index: int) -> Any:
        sample_path = self._sample_paths[index]
        sample = self.loader(sample_path)
        return sample

    def __len__(self) -> int:
        return len(self._sample_paths)

    def __repr__(self) -> str:
        s = ""
        s += f"{type(self).__name__}:\n"
        s += f"    Root directory: {self.root}\n"
        s += f"    Number of samples: {len(self)}\n"
        return s


class UnlabeledImageDataset(UnlabeledDataset):
    def __init__(
        self,
        root: FilePath,
        *,
        loader_mode: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        loader = PILLoader(loader_mode)
        super().__init__(root, loader)
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        sample = super().__getitem__(index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
