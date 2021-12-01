import functools
import gzip
from typing import Any, Iterable, Optional

import numpy as np

from ._parallel import execute_parallel
from ._typing import FilePath

__all__ = ["convert_binary_to_byte_seq", "convert_binary_to_byte_seq_parallel"]


def convert_binary_to_byte_seq(
    binary_file: FilePath,
    byte_seq_file: FilePath,
    *,
    length: int,
    padding: int = 256,
    compresslevel: int = 6,
) -> None:
    """Convert binary file to byte sequence file. (deprecated)"""

    with open(binary_file, "rb") as f:
        buffer = f.read(length)

    # Since PyTorch embedding layer requires int32/int64 as input, we have to
    # save each byte (plus padding) as int32 instead of int16.
    byte_seq: np.ndarray = np.frombuffer(buffer, dtype=np.uint8)
    byte_seq = byte_seq.astype(dtype=np.int32)

    padding_length = length - len(byte_seq)
    if padding_length > 0:
        padding_seq = np.full(padding_length, padding, dtype=np.int32)
        byte_seq = np.concatenate([byte_seq, padding_seq])

    # Compress byte sequence via gzip.
    with gzip.open(byte_seq_file, "wb", compresslevel=compresslevel) as f:
        f.write(byte_seq.tobytes())


def convert_binary_to_byte_seq_parallel(
    binary_files: Iterable[FilePath],
    byte_seq_files: Iterable[FilePath],
    *,
    length: int,
    padding: int = 256,
    n_jobs: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Convert binary file to byte sequence file in parallel. (deprecated)"""

    function = functools.partial(convert_binary_to_byte_seq, length=length, padding=padding)
    execute_parallel(function, binary_files, byte_seq_files, n_jobs=n_jobs, **kwargs)
