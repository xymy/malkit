from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..parallel import reduce_parallel
from ..typing import FilePath


class Loader:
    def __call__(self, path: FilePath) -> NDArray:
        raise NotImplementedError

    def parallel(self, paths: Iterable[FilePath], *, n_jobs: Optional[int] = None) -> NDArray:
        raise NotImplementedError

    def from_buffer(self, buffer: bytes) -> NDArray:
        raise NotImplementedError

    def __repr__(self) -> str:
        # Generate pretty formatted arguments.
        args = self._get_args()
        args_str = ", ".join(f"{a}={getattr(self, a)!r}" for a in args)
        return f"{type(self).__name__}({args_str})"

    def _get_args(self) -> Tuple[str, ...]:
        raise NotImplementedError


class MultiLoader(Loader):
    def __init__(self, loaders: List[Loader]) -> None:
        self.loaders = loaders

    def __call__(self, path: FilePath) -> NDArray:
        with open(path, "rb") as f:
            buffer = f.read()
        return self.from_buffer(buffer)

    def parallel(self, paths: Iterable[FilePath], *, n_jobs: Optional[int] = None) -> NDArray:
        return reduce_parallel(self, paths, n_jobs=n_jobs)

    def from_buffer(self, buffer: bytes) -> NDArray:
        return np.concatenate([loader.from_buffer(buffer) for loader in self.loaders])

    def _get_args(self) -> Tuple[str, ...]:
        args: List[str] = []
        for loader in self.loaders:
            args.extend(loader._get_args())
        return tuple(args)
