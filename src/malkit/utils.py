import functools
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ._parallel import execute_parallel
from ._typing import FilePath

__all__ = ["categorize_folders", "split_labels", "convert_bytes_to_binary", "convert_bytes_to_binary_parallel"]


def categorize_folders(root: FilePath, labels: pd.DataFrame, *, suffix: Optional[str] = None) -> bool:
    """Categorize samples and move them into class name folders."""

    root = Path(root)
    sample_names = [str(item) for item in labels.iloc[:, 0]]
    target_names = [str(item) for item in labels.iloc[:, 1]]

    for klass in set(target_names):
        cat_dir = root / klass
        if cat_dir.is_dir():
            return
        cat_dir.mkdir(parents=True, exist_ok=True)

    for sample_name, target_name in zip(sample_names, target_names):
        src = root / sample_name
        if suffix is not None:
            src = src.with_suffix(suffix)
        dst = root / target_name / sample_name
        src.rename(dst)


def split_labels(
    labels: pd.DataFrame, *, test_size: Optional[float] = None, train_size: Optional[float] = None, shuffle=True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split labels into two parts."""

    from sklearn.model_selection import train_test_split

    X = np.arange(len(labels))
    y = labels.iloc[:, 1].to_numpy()
    idx1, idx2 = train_test_split(X, test_size=test_size, train_size=train_size, shuffle=shuffle, stratify=y)
    return labels.iloc[idx1], labels.iloc[idx2]


def convert_bytes_to_binary(bytes_file: FilePath, binary_file: FilePath, *, qq: str = "00") -> None:
    """Convert bytes file to binary file."""

    with open(bytes_file, "r", encoding="ascii") as src, open(binary_file, "wb") as dst:
        for line in src:
            i = line.find(" ")
            if i < 0:
                raise ValueError(f"invalid bytes file {bytes_file!r}")
            data = line[i + 1 :].replace("??", qq)
            dst.write(bytes.fromhex(data))


def convert_bytes_to_binary_parallel(
    bytes_files: Iterable[FilePath],
    binary_files: Iterable[FilePath],
    *,
    qq: str = "00",
    n_jobs: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Convert bytes file to binary file in parallel."""

    function = functools.partial(convert_bytes_to_binary, qq=qq)
    execute_parallel(function, bytes_files, binary_files, n_jobs=n_jobs, **kwargs)
