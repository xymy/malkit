from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..typing import FilePath
from ._loader import Loader


class ClassifiedDataset:
    def __init__(
        self,
        root: FilePath,
        label: pd.DataFrame,
        loader: Loader,
    ) -> None:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"{root} is not a directory")

        self.root = root
        self.label = label
        self.loader = loader

        self._sample_names = [str(item) for item in label.iloc[:, 0]]
        self._target_names = [str(item) for item in label.iloc[:, 1]]

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

    def get_sample_path(self, index: int) -> Path:
        sample_name = self.sample_names[index]
        target_name = self.target_names[index]
        sample_path = self.root / target_name / sample_name
        return sample_path

    def get_sample_path_alter(self, index: int, root: FilePath) -> Path:
        sample_name = self.sample_names[index]
        target_name = self.target_names[index]
        sample_dir = Path(root) / target_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / sample_name
        return sample_path

    def get_sample(self, index: int) -> NDArray:
        sample_path = self.get_sample_path(index)
        sample = self.loader(sample_path)
        return sample

    def get_target(self, index: int) -> int:
        target_name = self.target_names[index]
        target = self.class_to_index[target_name]
        return target

    def load_X_y(self) -> Tuple[NDArray, NDArray]:
        X = []
        y = []
        for sample, target in self:
            X.append(sample)
            y.append(target)
        return np.stack(X), np.stack(y)

    def load_X_y_parallel(self, *, n_jobs: Optional[int] = None) -> Tuple[NDArray, NDArray]:
        paths = []
        y = []
        for index in range(len(self)):
            sample_name = self.sample_names[index]
            target_name = self.target_names[index]
            sample_path = self.root / target_name / sample_name
            target = self.class_to_index[target_name]
            paths.append(sample_path)
            y.append(target)
        X = self.loader.parallel(paths, n_jobs=n_jobs)
        return X, np.stack(y)

    def __iter__(self) -> Iterator[Tuple[NDArray, int]]:
        for index in range(len(self)):
            yield self.__getitem__(index)

    def __getitem__(self, index: int) -> Tuple[NDArray, int]:
        sample_name = self.sample_names[index]
        target_name = self.target_names[index]
        sample_path = self.root / target_name / sample_name
        sample = self.loader(sample_path)
        target = self.class_to_index[target_name]
        return sample, target

    def __len__(self) -> int:
        return len(self.sample_names)

    def __repr__(self) -> str:
        s = f"{type(self).__name__}:\n"
        s += f"    Root: {self.root}\n"
        s += f"    Loader: {self.loader}\n"
        s += f"    Number of samples: {len(self)}\n"
        s += f"    Number of classes: {len(self.index_to_class)}\n"
        return s
