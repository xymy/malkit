from typing import Any, Iterable, Optional

import numpy as np

from ._parallel import reduce_parallel
from ._typing import FilePath

__all__ = ["extract_byte_hist", "extract_byte_hist_parallel"]


def get_byte_hist(buffer: bytes) -> np.ndarray:
    """Get byte histogram."""

    byte_seq = np.frombuffer(buffer, dtype=np.uint8)
    return np.bincount(byte_seq, minlength=256)


def extract_byte_hist(binary_file: FilePath) -> np.ndarray:
    """Extract byte histogram."""

    with open(binary_file, "rb") as f:
        buffer = f.read()
    return get_byte_hist(buffer)


def extract_byte_hist_parallel(
    binary_files: Iterable[FilePath], *, n_jobs: Optional[int] = None, **kwargs: Any
) -> np.ndarray:
    """Extract byte histogram in parallel."""

    return reduce_parallel(extract_byte_hist, binary_files, n_jobs=n_jobs, **kwargs)
